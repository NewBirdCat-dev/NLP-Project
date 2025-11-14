"""
Resume Screening System (Using Best Pretrained Model)
-----------------------------------------------------
✅ Loads trained model (.joblib) — NO training
✅ Loads TF-IDF vectorizer & skill vocabulary (JSON)
✅ Extracts text from TXT / PDF resume
✅ Predicts Job Category
✅ Supports batch CSV resume classification (Relevant / Not Relevant)
"""

import os, re, json, joblib, tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
from difflib import get_close_matches
from PyPDF2 import PdfReader
import nltk
import pandas as pd

# NLTK setup
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------- FILE PATHS ----------
MODEL_DIR = "c:/Users/PMLS/Desktop/Newfolder/"
MODEL_PATH = MODEL_DIR + "best_resume_model.joblib"
TFIDF_PATH = MODEL_DIR + "tfidf_vectorizer.joblib"
SKILL_VOCAB_PATH = MODEL_DIR + "skill_vocab.json"


# ---------- TEXT PREPROCESS ----------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", str(text))
    text = re.sub(r"[^A-Za-z0-9#+\.\- ]", " ", text)
    return " ".join(text.split()).lower()

def preprocess_text(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    lemmatized = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 1]
    return " ".join(lemmatized)


# ---------- PDF TO TEXT ----------
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except:
            pass
    return text


# ---------- SKILL EXTRACTION ----------
def extract_skills(text, vocab):
    found = set()
    txt = text.lower()

    # Exact match
    for skill in vocab:
        if skill in txt:
            found.add(skill)

    # Fuzzy Match
    tokens = re.findall(r'\b[a-zA-Z+#]+\b', txt)
    for token in tokens:
        close = get_close_matches(token, vocab, cutoff=0.85)
        if close:
            found.add(close[0])

    return found


# ---------- PREDICT RESUME ----------
def predict_resume(text, model, tfidf, skill_vocab):
    clean = preprocess_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    return pred


# ---------- GUI APP ----------
class ResumeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Screening System")
        self.root.geometry("900x700")
        self.root.configure(bg="#e9eef7")

        tk.Label(root, text="📄 Resume Screening (Using Pretrained Model)",
                 font=("Arial", 18, "bold"), bg="#e9eef7").pack(pady=10)

        # Textbox for single resume
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
        self.text_area.pack(padx=10, pady=10)

        # Buttons
        tk.Button(root, text="Upload TXT / PDF Resume", command=self.load_file,
                  bg="#4CAF50", fg="white", width=25).pack(pady=5)
        tk.Button(root, text="Classify Resume", command=self.classify_resume,
                  bg="#2196F3", fg="white", width=25).pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"),
                                     bg="#e9eef7", fg="#222")
        self.result_label.pack(pady=10)

        # Batch button
        tk.Button(root, text="Upload CSV of Resumes", command=self.load_csv_batch,
                  bg="#ff5722", fg="white", width=25).pack(pady=5)

        self.load_model()


    # ---------- LOAD MODEL ----------
    def load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.tfidf = joblib.load(TFIDF_PATH)

            with open(SKILL_VOCAB_PATH, "r") as f:
                self.skill_vocab = set(json.load(f))

            print("✅ Loaded trained model and TF-IDF vectorizer")

        except Exception as e:
            messagebox.showerror("Error", f"Model Load Failed: {e}")


    # ---------- SINGLE RESUME FILE LOAD ----------
    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not path:
            return

        self.text_area.delete(1.0, tk.END)

        if path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        else:
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read()

        self.text_area.insert(tk.END, text)


    # ---------- CLASSIFY SINGLE RESUME ----------
    def classify_resume(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Paste or upload a resume first.")
            return

        prediction = predict_resume(text, self.model, self.tfidf, self.skill_vocab)
        skills = extract_skills(text, self.skill_vocab)

        self.result_label.config(
            text=f"🏷 Predicted Job Category: {prediction}\n🔧 Skills Found: {', '.join(skills) if skills else 'None'}"
        )


    # ---------- LOAD CSV ----------
    def load_csv_batch(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not path:
            return

        try:
            self.batch_df = pd.read_csv(path)
            text_col = [c for c in self.batch_df.columns if 'resume' in c.lower() or 'text' in c.lower()]

            if not text_col:
                messagebox.showerror("Error", "No resume/text column found in CSV.")
                return

            self.batch_text_col = text_col[0]

            # ✅ FIX: Get categories from trained model automatically
            categories = sorted(self.model.classes_)

            self.category_var = tk.StringVar()
            self.category_var.set(categories[0])

            self.category_menu = ttk.OptionMenu(self.root, self.category_var, categories[0], *categories)
            self.category_menu.pack(pady=5)

            tk.Button(self.root, text="Classify Batch CSV", command=self.classify_batch_csv,
                      bg="#f39c12", fg="white", width=25).pack(pady=5)

            messagebox.showinfo("CSV Loaded", f"✅ CSV loaded successfully with {len(self.batch_df)} resumes.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")


    # ---------- CLASSIFY CSV BATCH ----------
    def classify_batch_csv(self):
        if not hasattr(self, "batch_df"):
            messagebox.showwarning("Warning", "Load a CSV first.")
            return

        target_category = self.category_var.get()
        results = []

        for idx, row in self.batch_df.iterrows():
            text = str(row[self.batch_text_col])
            prediction = predict_resume(text, self.model, self.tfidf, self.skill_vocab)
            skills = extract_skills(text, self.skill_vocab)

            # ✅ FIX: partial matching (Data Science → Data Scientist)
            relevance = "Relevant" if target_category.lower() in prediction.lower() else "Not Relevant"

            preview = text[:50] + "..." if len(text) > 50 else text
            results.append((preview, relevance, prediction, ', '.join(skills) if skills else 'None'))

        # Display results in a new window
        result_win = tk.Toplevel(self.root)
        result_win.title("Batch Classification Results")
        result_win.geometry("1000x600")

        cols = ("Resume Preview", "Relevance", "Predicted Category", "Skills Found")
        tree = ttk.Treeview(result_win, columns=cols, show='headings')

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=250)

        tree.pack(fill=tk.BOTH, expand=True)

        for r in results:
            tree.insert("", tk.END, values=r)

        # Ask where to save
        export_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                   filetypes=[("CSV Files", "*.csv")])

        if export_path:
            out_df = pd.DataFrame(results, columns=cols)
            out_df.to_csv(export_path, index=False)
            messagebox.showinfo("Exported", f"✅ Results saved to {export_path}")


# ---------- RUN APP ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeApp(root)
    root.mainloop()
