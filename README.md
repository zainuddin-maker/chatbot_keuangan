
# 📈 Asisten Instrumen Keuangan (Financial Instruments AI Assistant)

Proyek ini adalah implementasi chatbot berbasis AI yang bertindak sebagai Asisten Virtual spesifik untuk topik instrumen keuangan, trading, dan investasi. Aplikasi ini dikembangkan sebagai **Final Project** untuk program **LLM-Based Tools and Gemini API Integration for Data Scientists** di **Hacktiv8**.

Aplikasi ini ditenagai oleh model multimodal Google Gemini melalui LangChain, dilengkapi dengan integrasi pencarian web secara *real-time* dan kemampuan membaca dokumen/gambar yang diunggah pengguna.

---

## 🎯 Use Case & Domain Pengetahuan

* **Kategori Bot:** *Financial & Investment Educational Assistant* (Asisten Edukasi Keuangan dan Investasi).
* **Domain Fokus:** Terbatas pada instrumen keuangan (saham, forex, kripto, komoditas, reksa dana, obligasi), analisis teknikal/fundamental, mekanisme pasar, dan manajemen risiko.
* **Guardrails (Batasan Sistem):**
  1. Chatbot secara otomatis akan menolak pertanyaan di luar topik investasi atau trading.
  2. Chatbot diinstruksikan untuk selalu menggunakan **Web Search Tool** jika mendeteksi pertanyaan terkait harga terkini, kurs, atau berita pasar hari ini.

> **⚠️ Disclaimer:** Chatbot ini dirancang khusus untuk tujuan **edukasi**. Chatbot ini **bukan penasihat keuangan berlisensi** dan tidak memberikan saran keputusan beli/jual secara pasti. Segala risiko keputusan finansial tetap berada di tangan pengguna.

---

## 🚀 Fitur Unggulan & Parameter Kreatif

Proyek ini menerapkan berbagai konfigurasi parameter kreatif dan fitur tingkat lanjut (RAG/Tool Use):

1. **Integrasi Tools (Real-time Web Search):**
   * Menggunakan LangChain Tools untuk memanggil pencarian internet setiap kali pengguna membutuhkan data harga atau berita terkini.
   * Mendukung dua *provider* pencarian: **DuckDuckGo** (Gratis/Default) dan **Exa AI** (Membutuhkan API Key khusus, hasil lebih mendalam).
2. **Multimodal & Ekstraksi Dokumen (File Upload):**
   * Pengguna dapat melampirkan berkas langsung ke dalam obrolan.
   * Mendukung dokumen teks (`.txt`, `.csv`, `.pdf` menggunakan `pypdf`) dan ekstraksi gambar / chart teknikal (`.png`, `.jpg`, `.jpeg`).
3. **Pengaturan Parameter AI Interaktif:**
   * **Temperature:** Slider (0.0 - 1.0) untuk mengatur tingkat formalitas dan variasi jawaban model.
   * **Max Output Tokens:** Pengguna dapat membatasi atau memperpanjang panjang maksimal jawaban model (128 - 2048 token).
4. **Manajemen Memori & UX Chat:**
   * Fitur **Edit Pesan:** Pengguna dapat mengedit pesan lama (termasuk lampiran) dan mengirim ulang percakapan dari titik tersebut.
   * Pembatasan riwayat (maksimal 20 pesan) yang dikirim ke LLM untuk optimasi penggunaan token tanpa menghapus tampilan riwayat di layar antarmuka pengguna.
   * *Log* pencarian mentah (*raw search result log*) ditampilkan dalam *expander* agar pengguna bisa memvalidasi sumber data asli.

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python
* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Orkestrasi LLM & Tools:** [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-google-genai`)
* **AI Model:** Google Gemini (`gemini-3.1-flash-lite`)
* **Eksternal API & Ekstraktor:**
  * `duckduckgo-search` (Pencarian web anonim/gratis)
  * `exa_py` (API Pencarian semantik)
  * `pypdf` (Pembaca dokumen PDF)

---

## 💻 Cara Menjalankan Aplikasi Secara Lokal

1. **Clone Repositori:**
   ```bash
   git clone [https://github.com/USERNAME/NAMA_REPOSITORI.git](https://github.com/USERNAME/NAMA_REPOSITORI.git)
   cd NAMA_REPOSITORI
   ```
