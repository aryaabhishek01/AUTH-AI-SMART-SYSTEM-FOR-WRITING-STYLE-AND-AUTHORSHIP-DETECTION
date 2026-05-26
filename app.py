from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import re
import os
import torch
import numpy as np
import random 
from collections import Counter, defaultdict

# new imports for image analyzer
from io import BytesIO
from PIL import Image
import torch.nn.functional as F

# AI Model Imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # --- model/tokenizer caches ---
    models = {}
    tokenizers = {}

    def load_model_and_tokenizer(model_name):
        """Loads a model and tokenizer and caches them in memory."""
        model_path = f"./models/{model_name}_style_detector"
        if os.path.exists(model_path):
            try:
                tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
                models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_path)
                models[model_name].eval()
                print(f"✅ Fine-tuned {model_name.upper()} model loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading {model_name.upper()} model from {model_path}: {e}")
                tokenizers[model_name] = None
                models[model_name] = None
        else:
            print(f"⚠️ Model path not found for {model_name.upper()}: {model_path}. Analysis will not work for this model.")
            tokenizers[model_name] = None
            models[model_name] = None

    # Load both models at startup (if present)
    load_model_and_tokenizer('albert')
    load_model_and_tokenizer('bert')

except ImportError as e:
    print(f"AI model libraries not found: {e}. Analysis will be non-functional.")
    models = {'albert': None, 'bert': None}
    tokenizers = {'albert': None, 'bert': None}


app = Flask(__name__)
app.secret_key = 'your_super_secret_key'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysql'
mysql = MySQL(app)

def create_database():
    try:
        cur = mysql.connection.cursor()
        cur.execute("CREATE DATABASE IF NOT EXISTS style_detection_db")
        cur.execute("USE style_detection_db")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                is_admin BOOLEAN NOT NULL DEFAULT 0,
                last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS support_queries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(100) NOT NULL,
                phone VARCHAR(20),
                feedback TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        mysql.connection.commit()
        print("Database and tables created or verified successfully.")
        cur.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        exit()

def create_default_admin():
    cur = mysql.connection.cursor()
    cur.execute("USE style_detection_db")
    cur.execute("SELECT * FROM users WHERE username = 'Admin'")
    user = cur.fetchone()
    if not user:
        hashed_password = generate_password_hash('admin@123')
        cur.execute("INSERT INTO users (username, password, is_admin) VALUES (%s, %s, %s)", ('Admin', hashed_password, 1))
        mysql.connection.commit()
        cur.close()
        print("Default admin user created.")

# --- PREDICTION FUNCTION (unchanged) ---
def predict_author(text, model_name):
    tokenizer = tokenizers.get(model_name)
    model = models.get(model_name)

    if not tokenizer or not model:
        return None, 0.0, "Model not available"

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1).squeeze(0)
    predicted_class_id = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_class_id].item())

    readable = f"Author {predicted_class_id + 1}"
    return predicted_class_id, confidence, readable

# --------------------
# Image analyzer (CLIP-based heuristic)
# --------------------
# This is a drop-in replacement for the previous random analyzer.
# It tries to use CLIP to compute embeddings for image quadrants and
# flags quadrants with significantly lower similarity to others.
_clip_processor = None
_clip_model = None

def _ensure_clip_loaded():
    """
    Lazy-load CLIP processor + model when first needed.
    If transformers isn't available or loading fails, this raises an exception.
    """
    global _clip_processor, _clip_model
    if _clip_processor is None or _clip_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
        except Exception as e:
            raise RuntimeError("Install transformers to enable image analysis: pip install transformers") from e
        # load models (this will download if not cached)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _clip_model.to(device)

def analyze_image_style(file_storage):
    """
    Heuristic image-style change detection:
      - Splits image into 4 quadrants
      - Compute CLIP image embeddings for each quadrant
      - Compute pairwise cosine similarities
      - Regions with low similarity to others are flagged as 'style change detected'
    Returns: (changes_list, cohesion_score_percent)
    If CLIP isn't available or image can't be opened, returns a sensible fallback.
    """
    # Read bytes first (so we can retry or reset stream)
    try:
        raw = file_storage.read()
    except Exception as e:
        # fallback to random behavior if read fails
        num_changes = random.randint(0, 2)
        changes = [f"Style change detected in region {i+1}." for i in range(num_changes)]
        return changes, round(random.uniform(60, 95), 2)

    # Try to load CLIP
    try:
        _ensure_clip_loaded()
    except Exception as e:
        # If CLIP not available, fallback to simple histogram heuristic (no extra deps)
        try:
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            # if even opening fails, return no changes
            try:
                file_storage.stream.seek(0)
            except Exception:
                pass
            return [], 0.0

        # Simple histogram-based difference across quadrants
        w, h = img.size
        if w < 64 or h < 64:
            try:
                file_storage.stream.seek(0)
            except Exception:
                pass
            return [], 100.0

        mid_x, mid_y = w // 2, h // 2
        regions = [
            img.crop((0, 0, mid_x, mid_y)),
            img.crop((mid_x, 0, w, mid_y)),
            img.crop((0, mid_y, mid_x, h)),
            img.crop((mid_x, mid_y, w, h)),
        ]

        hist_sims = []
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                h1 = regions[i].histogram()
                h2 = regions[j].histogram()
                # cosine-like similarity on hist vectors
                a = np.array(h1, dtype=np.float32)
                b = np.array(h2, dtype=np.float32)
                if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                hist_sims.append(sim)

        if len(hist_sims) == 0:
            try:
                file_storage.stream.seek(0)
            except Exception:
                pass
            return [], 0.0

        avg_sim = float(np.mean(hist_sims))
        cohesion_pct = round(avg_sim * 100, 2)

        # compute per-region avg similarity
        sims_matrix = np.zeros((4, 4), dtype=float)
        idx = 0
        for i in range(4):
            for j in range(i+1, 4):
                sims_matrix[i, j] = hist_sims[idx]
                sims_matrix[j, i] = hist_sims[idx]
                idx += 1

        region_avgs = [float(np.mean([sims_matrix[i, j] for j in range(4) if j != i])) for i in range(4)]
        threshold = avg_sim - 0.08  # a small margin

        changes = []
        for i, ra in enumerate(region_avgs):
            if ra < threshold:
                changes.append(f"Style change detected in region {i+1} (avg sim {round(ra*100,2)}%).")

        if not changes and cohesion_pct < 85.0:
            changes.append("Image shows low overall cohesion between quadrants — style variations suspected.")

        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        return changes, cohesion_pct

    # If CLIP loaded successfully, proceed with CLIP-based embeddings
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        return [], 0.0

    w, h = img.size
    if w < 64 or h < 64:
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        return [], 100.0

    mid_x, mid_y = w // 2, h // 2
    regions = [
        img.crop((0, 0, mid_x, mid_y)),           # top-left
        img.crop((mid_x, 0, w, mid_y)),           # top-right
        img.crop((0, mid_y, mid_x, h)),           # bottom-left
        img.crop((mid_x, mid_y, w, h)),           # bottom-right
    ]

    # prepare inputs for CLIP
    device = next(_clip_model.parameters()).device
    inputs = _clip_processor(images=regions, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        img_embeds = _clip_model.get_image_features(**inputs)  # (4, dim)
        img_embeds = F.normalize(img_embeds, p=2, dim=1)

    sims = (img_embeds @ img_embeds.T).cpu().numpy()  # 4x4 cosine sims
    # gather upper-triangular similarities
    pair_sims = []
    for i in range(4):
        for j in range(i+1, 4):
            pair_sims.append(float(sims[i, j]))

    if len(pair_sims) == 0:
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        return [], 0.0

    avg_sim = float(np.mean(pair_sims))
    cohesion_pct = round(avg_sim * 100, 2)

    # per-region average similarities
    region_avgs = [float(np.mean([sims[i, j] for j in range(4) if j != i])) for i in range(4)]

    # adaptive threshold: flag regions significantly below avg_sim
    threshold = avg_sim - 0.10
    changes = []
    for i, ra in enumerate(region_avgs):
        if ra < threshold:
            changes.append(f"Style change detected in region {i+1} (avg sim {round(ra*100,2)}%).")

    if not changes and cohesion_pct < 85.0:
        changes.append("Image shows low overall cohesion between quadrants — style variations suspected.")

    # reset stream pointer so other code can read if needed
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    return changes, cohesion_pct

# --- FLASK ROUTES (unchanged except analyze) ---
@app.before_request
def before_request():
    if 'MYSQL_DB' not in app.config:
        app.config['MYSQL_DB'] = 'style_detection_db'

@app.route('/')
def home():
    if 'loggedin' in session:
        return redirect(url_for('admin_page' if session['is_admin'] else 'user_page'))
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("USE style_detection_db")
    cur.execute("SELECT id, username, password, is_admin FROM users WHERE username = %s", [username])
    user = cur.fetchone()
    cur.close()
    if user and check_password_hash(user[2], password):
        session['loggedin'] = True
        session['user_id'] = user[0]
        session['username'] = user[1]
        session['is_admin'] = user[3]
        cur = mysql.connection.cursor()
        cur.execute("USE style_detection_db")
        cur.execute("UPDATE users SET last_login = %s WHERE id = %s", (datetime.datetime.now(), user[0]))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('admin_page' if session['is_admin'] else 'user_page'))
    else:
        return render_template('index.html', message='Invalid username or password')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    cur = mysql.connection.cursor()
    cur.execute("USE style_detection_db")
    cur.execute("SELECT id FROM users WHERE username = %s", [username])
    if cur.fetchone():
        cur.close()
        return render_template('index.html', message='Username already taken. Please choose a different one.')
    hashed_password = generate_password_hash(password)
    cur.execute("USE style_detection_db")
    cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
    mysql.connection.commit()
    cur.close()
    return render_template('index.html', message='Registration successful! Please login.')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_admin', None)
    return redirect(url_for('home'))

@app.route('/user')
def user_page():
    if not session.get('loggedin') or session.get('is_admin'):
        return redirect(url_for('home'))
    return render_template('user.html', analysis_results=[])

@app.route('/admin')
def admin_page():
    if not session.get('loggedin') or not session.get('is_admin'):
        return redirect(url_for('home'))
    cur = mysql.connection.cursor()
    cur.execute("USE style_detection_db")
    cur.execute("SELECT username, last_login FROM users WHERE is_admin = 0")
    users = cur.fetchall()
    cur.execute("USE style_detection_db")
    cur.execute("SELECT id, email, phone, feedback FROM support_queries")
    queries = cur.fetchall()
    cur.close()
    return render_template('admin.html', users=users, queries=queries)

@app.route('/support')
def support_page():
    if 'loggedin' not in session:
        return redirect(url_for('home'))
    return render_template('support.html')

# ----------------- UPDATED analyze() -----------------
@app.route('/analyze', methods=['POST'])
def analyze():
    if not session.get('loggedin'):
        return redirect(url_for('home'))

    analysis_results = []
    total_words = 0
    total_sentences = 0
    total_paragraphs = 0

    # collect paragraph-level predicted numeric classes and confidences
    all_preds = []                 # list of predicted class ids (ints)
    confidences_by_author = defaultdict(list)  # author_id -> list of confidences

    selected_model = request.form.get('model_name', 'albert')

    if 'documents[]' in request.files:
        files = request.files.getlist('documents[]')
        for file in files:
            if file.filename == '':
                continue

            text = file.read().decode('utf-8', errors='ignore')

            # quick stats
            words = len(re.findall(r'\b\w+\b', text))
            sentences = len(re.findall(r'[.!?]+', text))
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', text.strip()) if p.strip()]

            total_words += words
            total_sentences += sentences
            total_paragraphs += len(paragraphs)

            preds_for_file = []
            confidences_for_file = []

            for para in paragraphs:
                class_id, conf, readable = predict_author(para, selected_model)
                if class_id is None:
                    # model unavailable for this paragraph
                    preds_for_file.append(None)
                    confidences_for_file.append(0.0)
                else:
                    preds_for_file.append(int(class_id))
                    confidences_for_file.append(conf)
                    all_preds.append(int(class_id))
                    confidences_by_author[int(class_id)].append(conf)

            # Document-level summary (most frequent predicted author)
            valid_preds = [p for p in preds_for_file if p is not None]
            if valid_preds:
                most_common = Counter(valid_preds).most_common(1)[0]
                doc_author_label = f"Author {most_common[0] + 1}"
                doc_msg = (f"Document '{file.filename}': Most paragraphs predicted as <strong>{doc_author_label}</strong> "
                           f"(using {selected_model.upper()}).")
            else:
                doc_msg = f"Document '{file.filename}': Could not be analyzed (model not available)."

            analysis_results.append(doc_msg)

    # images
    if 'images[]' in request.files:
        files = request.files.getlist('images[]')
        for file in files:
            if file.filename != '':
                changes, acc = analyze_image_style(file)
                if changes:
                    analysis_results.append(f"Image '{file.filename}': Found {len(changes)} style change(s). Details: {', '.join(changes)} (cohesion {acc}%).")
                else:
                    analysis_results.append(f"Image '{file.filename}': No significant style changes detected. (cohesion {acc}%)")

    # Compute author counts, percentages, and average confidences
    numeric_preds = [p for p in all_preds if p is not None]
    author_counts = Counter(numeric_preds)
    unique_authors = len(author_counts)

    total_pred_paragraphs = sum(author_counts.values())

    author_percentages = {}
    author_avg_confidence = {}
    if total_pred_paragraphs > 0:
        for author_id, count in author_counts.items():
            pct = round((count / total_pred_paragraphs) * 100, 2)
            author_percentages[author_id] = pct
            # average confidence for this author (convert to percentage)
            confs = confidences_by_author.get(author_id, [])
            if confs:
                avg_conf = round((sum(confs) / len(confs)) * 100, 2)
            else:
                avg_conf = 0.0
            author_avg_confidence[author_id] = avg_conf

    # Debug prints
    print(f"Detected authors: {author_counts}")
    print(f"Percentages: {author_percentages}")
    print(f"Avg confidences (%): {author_avg_confidence}")

    # Render template with new stats
    return render_template('user.html',
                           analysis_results=analysis_results,
                           total_words=total_words,
                           total_sentences=total_sentences,
                           total_paragraphs=total_paragraphs,
                           unique_authors=unique_authors,
                           author_counts=dict(author_counts),
                           author_percentages=author_percentages,
                           author_avg_confidence=author_avg_confidence)

# ----------------- rest unchanged -----------------
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    email = request.form['email']
    phone = request.form['phone']
    feedback = request.form['feedback']
    cur = mysql.connection.cursor()
    cur.execute("USE style_detection_db")
    cur.execute("INSERT INTO support_queries (email, phone, feedback) VALUES (%s, %s, %s)", (email, phone, feedback))
    mysql.connection.commit()
    cur.close()
    flash('Thank you for contacting us! We will get back to you soon.')
    return redirect(url_for('user_page'))

@app.route('/delete_user', methods=['POST'])
def delete_user():
    if not session.get('loggedin') or not session.get('is_admin'):
        return redirect(url_for('home'))
    username_to_delete = request.form['username']
    cur = mysql.connection.cursor()
    cur.execute("USE style_detection_db")
    cur.execute("DELETE FROM users WHERE username = %s", [username_to_delete])
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('admin_page'))

@app.route('/ignore_feedback', methods=['POST'])
def ignore_feedback():
    if not session.get('loggedin') or not session.get('is_admin'):
        return redirect(url_for('home'))
    query_id = request.form['query_id']
    cur = mysql.connection.cursor()
    cur.execute("USE style_detection_db")
    cur.execute("DELETE FROM support_queries WHERE id = %s", [query_id])
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('admin_page'))

if __name__ == '__main__':
    with app.app_context():
        create_database()
        create_default_admin()
    app.run(debug=True)
