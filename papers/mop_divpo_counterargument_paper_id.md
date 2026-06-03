<div class="titlepage">

# Mixture of Persona dan Diverse Preference Optimization untuk Generasi Counter-Argument yang Beragam

**Makalah Utama Penelitian LLM**

**Dason Tiovino**  
NIM: 2310101002

Program Studi Informatika  
Universitas Pradita  
Tangerang  
2026

</div>

## Abstrak

Large Language Models (LLMs) semakin sering digunakan sebagai asisten pada tahap pre-writing. Namun, kemudahan ini membawa risiko penting: model dapat menghasilkan ide yang tampak fasih, tetapi secara konseptual saling mirip. Fenomena ini berhubungan dengan generative monoculture, yaitu kecenderungan model terarah untuk menghasilkan distribusi output yang sempit. Penelitian ini memfokuskan masalah tersebut ke domain yang lebih konkret, yaitu generasi counter-argument. Sistem yang dikembangkan mengombinasikan base model Qwen2.5-0.5B-Instruct dengan empat adapter LoRA berbasis persona kognitif: contrarian, systems thinker, cross-domain analogist, dan minimalist. Setiap persona dilatih melalui supervised fine-tuning, lalu dioptimalkan menggunakan Diverse Preference Optimization (DivPO). Evaluasi akhir membandingkan enam metode: base model, prompt-only, single LoRA, MoP-SFT, MoP+DivPO, dan MoP+DivPO v2. Hasil menunjukkan bahwa klaim penelitian harus dibuat secara hati-hati. Keempat persona terbukti sangat berbeda secara distribusional, dengan seluruh skor cosine SBERT antar persona sekitar 0.19-0.20. Namun, metode adapter SFT yang paling beragam justru menurunkan kualitas dan kegunaan output. DivPO v2 menjadi varian paling seimbang karena mempertahankan kualitas dan utility mendekati prompt-only, sambil memberi peningkatan kecil pada novelty dan lexical diversity. Dengan demikian, kontribusi utama penelitian ini bukan klaim bahwa DivPO menang di semua metrik, melainkan analisis empiris terhadap trade-off antara diversity dan quality dalam generasi counter-argument oleh LLM kecil.

## BAB 1. Pendahuluan

### 1.1 Latar belakang

LLM seperti GPT, Claude, Gemini, dan model terbuka lain telah mengubah cara manusia menulis, mencari ide, dan menyusun argumentasi. Pada tahap pre-writing, LLM sering digunakan untuk menghasilkan sudut pandang awal, counter-argument, outline, atau kemungkinan arah tulisan. Tahap ini penting karena ide awal sangat memengaruhi kualitas tulisan akhir. Jika ide awal generik, proses drafting dan editing berikutnya hanya akan memoles gagasan yang sejak awal sudah sempit.

Masalah utama yang diteliti di sini bukan kelancaran bahasa. Model modern sudah mampu menghasilkan teks yang fasih. Masalah yang lebih mendasar adalah keseragaman ide. Beberapa penelitian menunjukkan bahwa bantuan generative AI dapat meningkatkan kualitas atau kreativitas individu, tetapi pada saat yang sama menurunkan diversity kolektif dari konten yang dihasilkan [1, 2]. Dengan kata lain, banyak pengguna dapat berakhir pada ide yang mirip karena menerima saran dari model yang sama.

Wu et al. menyebut fenomena ini sebagai generative monoculture [3]. Model yang telah melalui alignment cenderung mengerucut ke respons yang aman, umum, dan disukai. Untuk tugas seperti menjawab pertanyaan faktual, konvergensi ini dapat berguna. Namun untuk pre-writing, konvergensi terlalu awal justru berbahaya karena penulis membutuhkan eksplorasi.

Proposal awal penelitian ini menggunakan framing "pre-writing ideation system" yang mengombinasikan Mixture of Persona (MoP) dan Diverse Preference Optimization (DivPO). Setelah eksperimen selesai, framing tersebut perlu diperjelas. Penelitian ini tidak mengevaluasi dampak langsung terhadap tulisan akhir manusia. Domain evaluasi yang benar adalah generasi counter-argument berbasis prompt CGA-CMV dan prompt sejenis. Counter-argument dipilih karena lebih mudah dinilai: sebuah output dapat diperiksa dari sisi relevansi, koherensi, substansi, novelty, utility, dan kesesuaian persona.

### 1.2 Rumusan masalah

Rumusan masalah penelitian ini adalah:

1. Apakah empat adapter persona kognitif menghasilkan distribusi output yang benar-benar berbeda, atau hanya variasi permukaan dari model yang sama?
2. Apakah DivPO dapat meningkatkan keseimbangan antara keberagaman dan kualitas dibandingkan base model, prompt-only, single LoRA, dan MoP-SFT?
3. Bagaimana hasil penelitian harus ditulis secara jujur ketika metrik diversity dan metrik quality tidak selalu bergerak ke arah yang sama?

### 1.3 Tujuan penelitian

Tujuan penelitian ini adalah:

1. Mengembangkan arsitektur MoP berbasis beberapa adapter LoRA untuk generasi counter-argument.
2. Menguji apakah persona yang dilatih memiliki output yang terpisah secara semantik.
3. Mengevaluasi pengaruh DivPO dan DivPO v2 terhadap diversity, quality, utility, novelty, dan persona fidelity.
4. Menyusun klaim penelitian yang koheren dengan hasil aktual, bukan hanya dengan proposal awal.

### 1.4 Posisi akhir penelitian

Posisi akhir penelitian ini bersifat lebih kuat karena lebih hati-hati. Hasil menunjukkan bahwa MoP berhasil pada aspek persona distinctness. Namun, hasil baseline tidak menunjukkan bahwa MoP+DivPO menang mutlak pada semua metrik. Metode SFT menghasilkan diversity paling tinggi, tetapi mengorbankan quality dan utility. Prompt-only tetap menjadi baseline yang kuat. MoP+DivPO v2 adalah varian paling seimbang karena menjaga kualitas dan kegunaan sambil mempertahankan sinyal diversity yang lebih baik daripada base model.

## BAB 2. Tinjauan Pustaka

### 2.1 LLM dan masalah diversity dalam writing assistance

LLM dilatih untuk memprediksi token berikutnya berdasarkan distribusi data yang sangat besar. Setelah pre-training, model biasanya melalui tahap alignment agar responsnya lebih aman, membantu, dan sesuai preferensi manusia. Alignment ini meningkatkan usability, tetapi dapat mengurangi variasi output. Untuk banyak tugas, hal ini tidak menjadi masalah. Untuk ideation dan writing assistance, hal ini menjadi krusial.

Doshi dan Hauser menunjukkan bahwa generative AI dapat meningkatkan kreativitas individu, tetapi menurunkan diversity kolektif dari konten yang dihasilkan [1]. Padmakumar dan He juga menunjukkan bahwa penggunaan language model dalam proses menulis dapat mengurangi content diversity [2]. Kedua hasil ini memperkuat alasan mengapa evaluasi sistem writing assistance tidak cukup hanya menggunakan kualitas teks.

Wu et al. memperkenalkan konsep generative monoculture [3]. Dalam konteks penelitian ini, generative monoculture berarti model cenderung memberi counter-argument yang aman dan umum, bukan counter-argument yang benar-benar membuka ruang berpikir baru bagi penulis.

### 2.2 Preference optimization dan DivPO

Metode preference optimization seperti RLHF dan DPO mengoptimalkan model agar lebih sering menghasilkan respons yang disukai. Namun, proses ini dapat mempertajam distribusi model. Jika respons yang paling disukai juga merupakan respons yang paling umum, maka model akan semakin terdorong ke output yang homogen.

DivPO mengubah cara pasangan preference dibentuk [5]. Alih-alih memilih respons paling berkualitas sebagai chosen response, DivPO memilih respons yang rare tetapi masih melewati ambang kualitas. Ide dasarnya sederhana: novelty tidak berguna jika responsnya buruk, tetapi quality saja tidak cukup jika semua respons mengulang framing yang sama.

Dalam penelitian ini, DivPO digunakan untuk memperluas distribusi output setiap persona. Varian DivPO v2 kemudian dikembangkan agar sinyal training lebih sesuai dengan evaluasi akhir. Jika evaluasi mengukur diversity antar persona, maka rarity dalam training juga harus dihitung terhadap kandidat lintas persona, bukan hanya kandidat dari persona yang sama.

### 2.3 LoRA, QLoRA, dan mixture-style adaptation

LoRA memungkinkan fine-tuning yang efisien dengan membekukan base model dan menambahkan matriks low-rank yang dapat dilatih [6]. QLoRA melanjutkan prinsip ini dengan quantization agar fine-tuning model besar dapat dilakukan dengan kebutuhan memori lebih kecil [7]. Teknik ini relevan karena penelitian ini menggunakan beberapa adapter kecil daripada melatih ulang seluruh model.

Pendekatan mixture-style seperti MixLoRA menggunakan beberapa expert berbasis LoRA dan mekanisme routing [8]. Penelitian ini mengambil prinsip tersebut, tetapi expert yang digunakan bukan expert domain tugas, melainkan persona kognitif. Tujuannya bukan hanya efisiensi komputasi, tetapi pemisahan gaya berpikir.

### 2.4 Mixture of Persona

Mixture of Persona dalam penelitian ini berarti model memiliki beberapa adapter yang masing-masing merepresentasikan cara berpikir berbeda. Empat persona yang digunakan adalah:

| Persona | Fungsi kognitif |
|---|---|
| Contrarian | Menantang asumsi tersembunyi dan posisi mayoritas |
| Systems thinker | Melihat sebab-akibat, constraint, feedback loop, dan efek tingkat dua |
| Cross-domain analogist | Meminjam mekanisme dari domain lain untuk membentuk analogi |
| Minimalist | Menghapus asumsi sekunder dan menyorot inti perbedaan pendapat |

Persona ini tidak dipilih hanya sebagai gaya bahasa. Masing-masing merepresentasikan operasi berpikir yang berguna dalam pre-writing. Contrarian membantu penulis melihat asumsi yang tidak dipertanyakan. Systems thinker membantu melihat struktur masalah. Cross-domain analogist membantu transfer ide lintas bidang. Minimalist membantu menemukan inti konflik tanpa distraksi.

## BAB 3. Metodologi

### 3.1 Arsitektur sistem

Base model yang digunakan adalah `Qwen/Qwen2.5-0.5B-Instruct`. Model dasar dibekukan, lalu empat adapter LoRA dilatih di atasnya. Setiap adapter memiliki system prompt dan dataset yang disesuaikan dengan persona.

| Persona | Sumber data utama | Alasan pemilihan |
|---|---|---|
| Contrarian | Conversations Gone Awry - ChangeMyView | Mengandung pola debat dan bantahan terhadap klaim |
| Systems thinker | StackExchange question-answering | Banyak jawaban menjelaskan constraint, sebab, dan solusi struktural |
| Cross-domain analogist | arXiv abstracts | Mengandung mekanisme abstrak yang dapat dipindahkan lintas domain |
| Minimalist | IBM argument quality ranking | Mengandung argumen singkat yang dapat dipakai untuk kritik ringkas |

Prinsip yang digunakan adalah cognitive fingerprint: data fine-tuning harus mencerminkan operasi berpikir yang ingin dipelajari model. Jika dataset tidak membawa pola kognitif yang tepat, adapter hanya akan mempelajari gaya permukaan.

### 3.2 Tahap training

Training dilakukan dalam dua tahap. Tahap pertama adalah supervised fine-tuning untuk setiap persona. Tahap kedua adalah DivPO. Pada DivPO awal, rarity dihitung terhadap kandidat dalam persona yang sama. Pada DivPO v2, rarity dihitung terhadap seluruh kandidat lintas persona pada prompt yang sama.

Perubahan penting pada DivPO v2 adalah penghapusan prompt-response cosine sebagai komponen quality proxy. Untuk counter-argument, respons yang baik tidak selalu dekat secara semantik dengan prompt. Respons yang baik dapat menolak, membalik, atau mereframing klaim. Karena itu, DivPO v2 menggunakan quality floor berbasis koherensi dan panjang, lalu memilih kandidat yang rare di antara kandidat lintas persona.

### 3.3 Metode pembanding

Evaluasi akhir menggunakan enam metode:

| Metode | Deskripsi |
|---|---|
| Base | Qwen2.5-0.5B-Instruct tanpa adapter dan tanpa persona prompt |
| Prompt-only | Base model dengan instruksi persona di system prompt |
| Single LoRA | Satu adapter dilatih pada gabungan semua data persona |
| MoP-SFT | Empat adapter persona setelah supervised fine-tuning |
| MoP+DivPO | Empat adapter persona setelah DivPO awal |
| MoP+DivPO v2 | Empat adapter persona setelah DivPO lintas persona |

Untuk metode berbasis persona, setiap prompt menghasilkan empat output, masing-masing dari satu persona. Untuk base model, setiap prompt menghasilkan empat output stokastik tanpa persona.

### 3.4 Evaluasi

Evaluasi dibagi menjadi dua tahap besar.

Pertama, persona distinctness validation. Tahap ini menggunakan 20 prompt netral. Untuk setiap prompt dan setiap persona, model menghasilkan lima output. Total output adalah 400. Semua output di-embed menggunakan `all-MiniLM-L6-v2`, lalu dihitung cosine similarity antar persona.

Kedua, baseline evaluation. Tahap ini menggunakan 30 prompt dan enam metode. Setiap metode menghasilkan 120 output, sehingga totalnya 720 output. Metrik otomatis yang digunakan adalah Self-BLEU, Distinct-1, Distinct-2, dan SBERT cosine. Selain itu, digunakan LLM-as-judge untuk quality, novelty, pre-writing utility, dan persona fidelity.

## BAB 4. Hasil dan Pembahasan

### 4.1 Hasil persona distinctness

Hasil persona distinctness adalah sebagai berikut:

| Pasangan persona | SBERT cosine |
|---|---:|
| Contrarian - Systems thinker | 0.2046 |
| Contrarian - Cross-domain analogist | 0.2030 |
| Systems thinker - Cross-domain analogist | 0.1973 |
| Contrarian - Minimalist | 0.1922 |
| Systems thinker - Minimalist | 0.1906 |
| Cross-domain analogist - Minimalist | 0.1904 |

Seluruh pasangan berada jauh di bawah threshold 0.5. Tidak ada pasangan yang mendekati threshold collapse 0.7. Ini adalah bukti paling kuat untuk klaim arsitektur MoP. Keempat persona tidak collapse menjadi satu distribusi yang sama.

### 4.2 Hasil baseline evaluation

| Method | Self-BLEU lower | Dist-1 higher | Dist-2 higher | SBERT lower | Quality higher | Novelty higher | Utility higher | Fidelity higher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.086 | 0.481 | 0.894 | 0.675 | 3.417 | 3.192 | 3.058 | - |
| Prompt-only | 0.062 | 0.576 | 0.923 | 0.708 | 3.975 | 3.458 | 3.542 | 3.042 |
| Single LoRA | 0.040 | 0.597 | 0.945 | 0.481 | 2.589 | 4.650 | 2.308 | 1.975 |
| MoP-SFT | 0.026 | 0.559 | 0.943 | 0.488 | 2.633 | 4.700 | 2.308 | 2.192 |
| MoP+DivPO | 0.056 | 0.556 | 0.920 | 0.705 | 3.931 | 3.642 | 3.542 | 3.025 |
| MoP+DivPO v2 | 0.051 | 0.580 | 0.931 | 0.690 | 3.956 | 3.650 | 3.550 | 3.058 |

Tabel ini menunjukkan bahwa hasil penelitian tidak boleh disederhanakan menjadi "MoP+DivPO selalu menang." Ada trade-off yang jelas.

Pertama, base model memiliki performa lebih rendah daripada metode berbasis persona pada sebagian besar metrik judge dan lexical diversity. MoP+DivPO memiliki peningkatan signifikan terhadap base pada Self-BLEU, Distinct-1, Distinct-2, quality, novelty, dan utility berdasarkan tabel signifikansi yang tersedia.

Kedua, Single LoRA dan MoP-SFT menghasilkan diversity paling tinggi. Nilai Self-BLEU dan SBERT mereka rendah, sedangkan novelty dari judge tinggi. Namun, kualitas dan kegunaannya turun drastis. MoP-SFT memiliki quality 2.633 dan utility 2.308, sedangkan prompt-only memiliki quality 3.975 dan utility 3.542. Artinya, diversity yang tinggi tidak otomatis berarti output lebih berguna.

Ketiga, MoP+DivPO dan MoP+DivPO v2 berada pada posisi seimbang. Mereka tidak mengalahkan SFT pada semantic diversity, tetapi berhasil memulihkan quality dan utility. Dibandingkan MoP+DivPO awal, v2 lebih baik pada Distinct-1, Distinct-2, SBERT cosine, quality, novelty, utility, dan fidelity. Karena itu, v2 adalah varian final yang paling layak ditekankan.

### 4.3 Kalibrasi LLM-as-judge

Hasil inter-judge agreement menunjukkan:

| Metrik | Spearman rho | Interpretasi |
|---|---:|---|
| Quality | 0.6972 | Borderline acceptable |
| Persona fidelity | 0.6200 | Borderline acceptable |
| Novelty | 0.2659 | Low agreement |
| Utility | 0.5546 | Low-to-borderline agreement |

Implikasinya penting. Klaim novelty dari judge harus dibaca secara hati-hati karena agreement rendah. Quality dan fidelity lebih stabil. Oleh karena itu, novelty sebaiknya dipakai sebagai indikator arah, bukan bukti tunggal.

### 4.4 Temuan utama

**Temuan 1: Persona separation berhasil.** Keempat adapter menghasilkan distribusi output yang berbeda. Ini mendukung klaim bahwa MoP memberi inter-persona diversity.

**Temuan 2: Diversity tanpa quality constraint tidak cukup.** Single LoRA dan MoP-SFT menghasilkan output yang sangat beragam, tetapi kualitas dan utility rendah. Untuk pre-writing, output yang berbeda tetapi lemah tidak membantu penulis.

**Temuan 3: DivPO v2 adalah mekanisme balancing.** DivPO v2 bukan pemenang mutlak pada semua metrik. Kontribusinya adalah menjaga kualitas dan utility sambil mempertahankan sebagian keuntungan diversity.

**Temuan 4: Prompt-only adalah baseline kuat.** Prompt-only hampir menyamai atau mengalahkan v2 pada quality. Ini harus diakui. Keunggulan sistem terlatih perlu dijelaskan melalui distinctness, controllability, dan potensi skalabilitas, bukan hanya angka agregat.

### 4.5 Pembahasan kritis

Hasil penelitian ini menunjukkan bahwa diversity tidak bisa diperlakukan sebagai satu angka tunggal. Ada lexical diversity, semantic diversity, dan cognitive-style diversity. Distinct-1/2 dan Self-BLEU melihat variasi permukaan. SBERT melihat kedekatan makna. Persona distinctness melihat apakah gaya berpikir antar adapter benar-benar berbeda.

Metode SFT bergerak kuat ke arah diversity, tetapi menurunkan usability. Ini kemungkinan terjadi karena model kecil dan adapter terbatas dapat didorong ke wilayah output yang tidak biasa, tetapi belum cukup stabil untuk menjaga kualitas argumen. DivPO v2 memperbaiki hal ini dengan memilih kandidat rare yang tetap melewati quality floor dan dengan menghitung rarity terhadap pool lintas persona.

Secara naratif, paper ini paling koheren jika dibingkai sebagai studi diversity-quality frontier. Base model berada di area menengah. Prompt-only berada pada area high usability. SFT berada pada area high diversity tetapi low usability. DivPO v2 berada pada area balanced: tidak seekstrem SFT dalam diversity, tetapi jauh lebih berguna.

## BAB 5. Kesimpulan dan Keterbatasan

### 5.1 Kesimpulan

Penelitian ini awalnya diusulkan sebagai sistem pre-writing ideation berbasis Mixture of Persona dan DivPO. Setelah hasil eksperimen tersedia, framing yang lebih tepat adalah generasi counter-argument yang beragam menggunakan cognitive persona adapters dan diversity-aware preference optimization.

Hasil paling kuat adalah validasi persona distinctness. Semua pasangan persona memiliki SBERT cosine rendah, sehingga empat persona dapat dipertahankan. Hasil kedua adalah trade-off antara diversity dan quality. Adapter SFT menghasilkan diversity tinggi, tetapi menurunkan quality dan utility. Hasil ketiga adalah peran DivPO v2 sebagai metode balancing. DivPO v2 tidak menang mutlak, tetapi menjadi varian paling seimbang untuk ditekankan dalam paper.

Klaim akhir yang paling defensible adalah: MoP berhasil memisahkan mode berpikir, dan DivPO v2 membantu menjaga keseimbangan antara keberagaman dan kegunaan dalam generasi counter-argument oleh LLM kecil.

### 5.2 Keterbatasan

Pertama, model dasar yang digunakan masih kecil, yaitu Qwen2.5-0.5B-Instruct. Beberapa kelemahan dapat berasal dari kapasitas model, bukan dari arsitektur MoP atau DivPO itu sendiri.

Kedua, evaluasi menggunakan LLM-as-judge. Walaupun praktis, agreement untuk novelty masih rendah. Penelitian lanjutan sebaiknya menambahkan human evaluation.

Ketiga, penelitian ini belum mengukur dampak langsung terhadap kualitas tulisan akhir manusia. Pre-writing adalah motivasi, sedangkan evaluasi aktual adalah counter-argument generation.

Keempat, dataset persona berasal dari domain berbeda. Ini membantu pemisahan style, tetapi juga dapat membawa artifact domain. Dataset counter-argument yang lebih bersih untuk semua persona akan memperkuat penelitian berikutnya.

### 5.3 Arah penelitian lanjutan

Penelitian berikutnya dapat menggunakan base model yang lebih besar, memperbaiki data counter-argument lintas persona, menambahkan evaluasi manusia, dan menguji apakah output MoP+DivPO benar-benar membantu penulis menghasilkan tulisan akhir yang lebih beragam. Selain itu, DivPO v2 dapat dikembangkan dengan reward model kualitas yang lebih kuat daripada heuristic quality floor.

## Daftar Pustaka

1. A. R. Doshi dan O. P. Hauser, "Generative AI enhances individual creativity but reduces the collective diversity of novel content," *Science Advances*, 2024. https://www.science.org/doi/10.1126/sciadv.adn5290
2. V. Padmakumar dan H. He, "Does Writing with Language Models Reduce Content Diversity?," ICLR, 2024. https://arxiv.org/abs/2309.05196
3. F. Wu, E. Black, dan V. Chandrasekaran, "Generative Monoculture in Large Language Models," ICLR, 2025. https://arxiv.org/abs/2407.02209
4. R. Kirk et al., "Understanding the Effects of RLHF on LLM Generalisation and Diversity," ICLR, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/5a68d05006d5b05dd9463dd9c0219db0-Abstract-Conference.html
5. J. Lanchantin et al., "Diverse Preference Optimization," arXiv, 2025. https://arxiv.org/abs/2501.18101
6. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR, 2022. https://openreview.net/forum?id=nZeVKeeFYf9
7. T. Dettmers, A. Pagnoni, A. Holtzman, dan L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS, 2023. https://arxiv.org/abs/2305.14314
8. D. Li et al., "MixLoRA: Enhancing Large Language Model Fine-Tuning with LoRA-based Mixture of Experts," arXiv, 2024. https://arxiv.org/abs/2404.15159
9. J. Zhang et al., "Conversations Gone Awry: Detecting Early Signs of Conversational Failure," ACL, 2018. https://arxiv.org/abs/1805.05345
10. S. Gretz et al., "A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis," AAAI, 2020. https://arxiv.org/abs/1911.11408
11. Qwen Team, "Qwen2.5-0.5B-Instruct," Hugging Face model card, 2024. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
