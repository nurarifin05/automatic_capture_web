<h2># automatic_capture_web</h2>
<h4>Halo semua, ini adalah tools yang digunakan untuk mengambil gambar dari sebuah website.
Tools ini dapat digunakan sebagai alternatif dari tools seperti gowitness dan sejenisnya agar dapat berjalan yang lebih ringan.</h4>
Ikuti langkah-langkah berikut untuk meng-clone, menginstal dependensi, dan menjalankan proyek:

```bash
# Clone repository
git clone https://github.com/username/nama-repository.git
cd nama-repository

# Install dependencies
pip install -r requirements.txt  # atau gunakan pip3 jika perlu

# Install Playwright (untuk otomatisasi browser)
playwright install

# Buat folder dan file input
mkdir -p input
touch input/domain.txt

# Jalankan program
python3 main.py
