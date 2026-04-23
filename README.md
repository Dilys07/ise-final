# Lab 2: Configuration Performance Learning
## Dataset details
This lab contains datasets for nine widely studied configurable systems, each featuring different option types (e.g., integers, booleans, and enumerations), application domains, performance objectives, and programming languages.

The repository has nine folders, one for each system. Inside each folder, you'll find multiple CSV files, each recording configuration and performance data under a specific workload. Particularly:

- The first ***n-1* columns** represent the **configuration options**, and the **rightmost column** represents the **performance metric**;
- Each **row** corresponds to one **measurement sample** of configuration and performance.

Below is a detailed introduction to each system.

---

### **JUMP3R**
JUMP3R is a Java-based audio encoding system designed for efficient and high-performance audio processing. Audio encoding involves converting audio data into specific formats, often to reduce size while maintaining quality.  
- **Version:** 1.0.4  
- **Configuration Options:** 16  
- **Total Configurations:** 4,196  
- **Performance**: Runtime performance in audio processing tasks.
- **Workloads Tested:**  
  - `beethoven.wav` (classical music)  
  - `dual-channel.wav` (stereo encoding)  
  - `helix.wav` (synthesized tones)  
  - `single-channel.wav` (mono audio)  
  - `speech.wav` (voice data)  
  - `sweep.wav` (frequency sweeps)  

---

### **KANZI**
KANZI is a Java-based file compression utility aimed at reducing file sizes for storage and transmission. It balances compression speed and ratio, making it suitable for a wide range of use cases.  
- **Version:** 1.9  
- **Configuration Options:** 24  
- **Total Configurations:** 4,112  
- **Performance**: Runtime efficiency in compressing files.
- **Workloads Tested:**  
  - Nine files: `ambivert`, `artificl`, `deepfield`, `enwik8`, `fannie_mae_500k`, `large`, `misc`, `silesia`, `vmlinux`


---

### **DCONVERT**
DCONVERT is a Java-based image scaling tool optimized for resizing images while maintaining visual fidelity. It is widely used in graphics processing and digital content creation.  
- **Version:** 1.0.0-a7  
- **Configuration Options:** 18  
- **Total Configurations:** 6,764  
- **Performance**: Runtime speed for scaling operations.
- **Workloads Tested:**  
  - Image formats: JPEG, PNG, PSD, and SVG  
  - Sizes: Small, Medium, and Large  
  - Examples: `jpeg-small`, `png-medium`, `psd-large`, `svg-large`  

---

### **H2**
H2 is a Java-based relational database system optimized for throughput, suitable for embedded systems and development environments. It focuses on fast data transactions and efficient query execution.  
- **Version:** 1.4.200  
- **Configuration Options:** 16  
- **Total Configurations:** 1,954  
- **Performance**: Throughput in database operations.
- **Workloads Tested:**  
  - Financial: `smallbank-1`, `smallbank-10`  
  - E-commerce: `tpcc-2`, `tpcc-8`  
  - Voting systems: `voter-2`, `voter-16`  
  - Cloud-scale datasets: `ycsb-600`, `ycsb-2400`  

---

### **BATLIK**
BATLIK is a Java-based SVG rasterizer that converts vector graphics into raster images. It is widely used in rendering scalable graphics into pixel-based formats for display or printing.  
- **Version:** 1.14  
- **Configuration Options:** 10  
- **Total Configurations:** 1,919 
- **Performance**: Runtime performance in rasterizing vector graphics.
- **Workloads Tested:**  
  - 11 SVG datasets: `corona`, `cranium`, `cubus`, etc.
 

---

### **XZ**
XZ is a C/C++ file compression tool that utilizes the LZMA2 compression algorithm, achieving a high compression ratio. It is popular for archiving and data transfer applications.  
- **Version:** 5.2.0  
- **Configuration Options:** 33  
- **Total Configurations:** 1,999  
- **Performance**: Runtime efficiency in compressing files.
- **Workloads Tested:**  
  - 13 files:`ambivert.wav.tar`, `deepfield.tar`, `uiq2-16.bin`, etc.

---

### **LRZIP**
LRZIP is a C/C++ file compression utility designed for high compression ratios with long-range redundancy detection, particularly effective for large files with repeating patterns.  
- **Version:** 0.651  
- **Configuration Options:** 11  
- **Total Configurations:** 190  
- **Performance**: Runtime in large file compression.
- **Workloads Tested:**  
  - 13 datasets, overlapping with XZ. 

---

### **X264**
X264 is a video encoding system written in C/C++ that implements the H.264/MPEG-4 AVC standard. It focuses on high-quality encoding with efficient bitrate reduction.  
- **Version:** `baee400`  
- **Configuration Options:** 25  
- **Total Configurations:** 3,113 
- **Performance**: Runtime in encoding videos.
- **Workloads Tested:**  
  - Nine video files: `blue_sky_1080p25_short.y4m`, `Johnny_1280x720_60_short.y4m`, etc.


---

### **Z3**
Z3 is a C/C++ SMT (Satisfiability Modulo Theories) solver used for automated theorem proving and formal verification. It is widely applied in software verification and symbolic computation.  
- **Version:** 4.8.14  
- **Configuration Options:** 12  
- **Total Configurations:** 1,011  
- **Performance**: Runtime for solving logical formulas.
- **Workloads Tested:**  
  - 12 logical problems: `AUFNIRA_z3.637557.smt2`, `QF_LIA_tightrhombus`, etc. 






