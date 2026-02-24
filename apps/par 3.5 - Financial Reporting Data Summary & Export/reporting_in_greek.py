import streamlit as st
import fitz
import requests
import concurrent.futures
import time

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Σύνοψη Οικονομικών Αρχείων", layout="wide")
st.title("Σύνοψη/Ανάλυση Οικονομικών Δεδομένων")
st.write("Ταχύτερη επεξεργασία μέσω βελτιστοποιημένων κλήσεων Ollama.")

# -------------------------------
# Language & Mode Selector
# -------------------------------
language = st.selectbox("🌍 Γλώσσα Περίληψης:", ["Αγγλικά", "Ελληνικά", "Ισπανικά", "Γαλλικά"], index=1)
mode = st.radio("⚡ Επιλογή Λειτουργίας:", ["Αναλυτική", "Γρήγορη"], index=0)

# Εμφάνιση περιγραφής ανάλογα με τη λειτουργία
if mode == "Αναλυτική":
    st.info("🧠 *Αναλυτική Λειτουργία:* Επεξεργάζεται **όλο το PDF** με μεγαλύτερα τμήματα και περισσότερη λεπτομέρεια. "
            "Ιδανική για τελική σύνοψη και πλήρη ανάλυση. (Πιο αργή αλλά πιο ακριβής)")
else:
    st.warning("⚡ *Γρήγορη Λειτουργία:* Επεξεργάζεται **μόνο τα πρώτα 3 τμήματα** του PDF για μια συνοπτική εκτίμηση. "
               "Ιδανική για γρήγορο preview ή μεγάλα αρχεία. (Πιο γρήγορη αλλά λιγότερο αναλυτική)")

# -------------------------------
# Parameters based on mode
# -------------------------------
if mode == "Αναλυτική":
    CHUNK_SIZE = 10000
    MAX_WORKERS = 6
elif mode == "Γρήγορη":
    CHUNK_SIZE = 5000
    MAX_WORKERS = 2

# -------------------------------
# LLaMA3 call
# -------------------------------
def llama3_ollama(prompt, model="llama3", timeout=10000):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"❌ Σφάλμα κατά την κλήση Ollama: {e}"

# -------------------------------
# Extract text from PDF
# -------------------------------
@st.cache_data(show_spinner="📄 Εξαγωγή κειμένου από PDF...")
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return " ".join(page.get_text() for page in doc)

# -------------------------------
# Split text into chunks
# -------------------------------
def chunk_text(text, chunk_size=5000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# -------------------------------
# Summarize chunk
# -------------------------------
def summarize_chunk(chunk, language):
    prompt = f"Σύνοψη στα {language}:\n\n{chunk}"
    return llama3_ollama(prompt)

# -------------------------------
# Summarize text with progress and total timing
# -------------------------------
def summarize_text(text):
    start_total = time.time()  # ⏱️ Έναρξη συνολικής μέτρησης

    chunks = chunk_text(text, CHUNK_SIZE)
    
    if mode == "Γρήγορη" and len(chunks) > 3:
        chunks = chunks[:3]
        st.warning("⚡ Fast Mode: χρησιμοποιούνται μόνο τα πρώτα 3 chunks για γρήγορη εκτίμηση.")
    
    st.info(f"🔍 Διαχωρίστηκαν σε {len(chunks)} τμήματα ({mode} mode).")

    summaries = []
    progress_bar = st.progress(0)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(summarize_chunk, chunk, language): idx for idx, chunk in enumerate(chunks)}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            summaries.append(result)
            progress_bar.progress((i + 1) / len(chunks))
            st.write(f"✅ Ολοκληρώθηκε τμήμα {i + 1}/{len(chunks)}")

    # Τελική συνδυαστική περίληψη
    final_prompt = (
        f"Συνδύασε τις παρακάτω περιλήψεις και γράψε μια συνολική περίληψη στα {language}:\n\n"
        + "\n\n".join(summaries)
    )
    final_summary = llama3_ollama(final_prompt)

    total_elapsed = time.time() - start_total  # ⏱️ Συνολικός χρόνος
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    st.subheader("🧠 Τελική Περίληψη")
    st.write(final_summary)
    st.success(f"🏁 Χρόνος περίληψης: {minutes}m {seconds}s ({total_elapsed:.2f} sec)")

    return final_summary

# -------------------------------
# PDF Upload
# -------------------------------
pdf_file = st.file_uploader("📂 Ανέβασε PDF Αναφορά", type=["pdf"])

if pdf_file is not None:
    # ⏱️ Έναρξη συνολικού χρόνου όλων των βημάτων
    start_total = time.time()

    # Εξαγωγή κειμένου
    text = extract_text_from_pdf(pdf_file.read())
    st.write(f"Μήκος κειμένου: {len(text)} χαρακτήρες")
    st.success(f"✅ Εξήχθη κείμενο ({len(text)} χαρακτήρες)")

    # Περίληψη
    st.header("📄 Περίληψη Αναφοράς")
    summary = summarize_text(text)

    # Βασικά Οικονομικά Μεγέθη
    st.header("📊 Βασικά Οικονομικά Μεγέθη")
    metrics_prompt = f"""
    Από το κείμενο, δώσε συνοπτικά:
    - Όγκοι Πωλήσεων 9Μ24 και σύγκριση με το 2023
    - Πωλήσεις 9Μ24 και σύγκριση με το 2023
    - EBITDA 9Μ24 και σύγκριση με το 2023
    - Καθαρά Κέρδη 9Μ24 και σύγκριση με το 2023
    - Απασχολούμενα κεφάλαια 9Μ24 και σύγκριση με το 2023
    - Καθαρός Δανεισμός 9Μ24 και σύγκριση με το  2023
    - Προσωρινό μέρισμα ανά μετοχή στους μετόχους
    - Επιπτώσεις των τιμών του πετρελαίου
    - Ζήτηση των καυσίμων στην εσωτερική αγορά

    Γράψε στα {language}.
    Κείμενο:
    {text[:8000]}
    """
    metrics = llama3_ollama(metrics_prompt)
    st.write(metrics)

    # Insights
    st.header("⚡ Στρατηγική")
    insights_prompt = f"""
    Από το κείμενο, δώσε:
    - Κύριες εξελίξεις γενικά και σε κάθε στρατηγικό κλάδο

    Γράψε στα {language}.
    Κείμενο:
    {text[:8000]}
    """
    insights = llama3_ollama(insights_prompt)
    st.write(insights)

    # ⏱️ Τέλος συνολικού χρόνου
    end_total = time.time()
    total_elapsed = end_total - start_total

    # Εμφάνιση συνολικού χρόνου (λεπτά:δευτερόλεπτα)
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)
    st.success(f"🏁 Συνολικός χρόνος επεξεργασίας: {minutes}m {seconds}s ({total_elapsed:.2f} sec)")
