# Handoff — Debt Closure rantai #82 (spec → 13 tiket (+2 pecahan) → 3 PR)

> Untuk agent penerus. Dokumen ini ringkasan **state dan konteks**, bukan duplikasi spec.
> Baca artifact yang dirujuk, jangan tebak.
>
> **Sudah diverifikasi terhadap kode** pada `main` @ `4566fae` via codegraph (`codegraph_explore`):
> nama simbol, nomor baris, varian enum, jumlah baris gate, dan jumlah `#[ignore]` dicek langsung ke
> sumber. Tujuh ketidakakuratan versi pertama sudah dikoreksi (§0, §1, §2, §3, §4a, §4d). Bila
> handoff ini dan kode berbeda, **kode yang benar** — rekam koreksinya di sini.

## 0. Posisi sekarang (2026-10-19)

- `main` @ **`7415e77`** (merge PR #102), worktree bersih, tanpa tiket/branch tiket tersisa.
- **Tidak ada PR terbuka.**
- Rantai #82: **14 dari 15 sub-issue CLOSED** — 13 tiket asli #83–#95 (semua closed) + #100
  (closed); #95 dipecah menjadi #100 + #101. Sisa satu: **#101** (`ready-for-human`).
- Sisa issue lain: **#99** (`bug`, CI merah permanen).

| Issue | Label | Arti |
|---|---|---|
| #82 | `ready-for-agent` | Spec induk, tetap OPEN sampai #101 selesai |
| #101 | `ready-for-human` | Ukur F12/F15 di Mac dengan Xcode penuh + GGUF asli |
| #99 | `bug` | CI `main` merah karena objc2 Metal panic di runner |

## 1. Flow yang sudah dijalankan

```text
#82 spec ──► 13 tiket ──► 3 PR
   │
   ├── PR #96  (MERGED e354663)  #83 #84 #85 #93
   ├── PR #98  (MERGED 0b07121)  #86 #87 #88 #89 #90 #91 #92 #94
   └── PR #102 (MERGED 7415e77)  #100  (pecahan #95)
```

Total commit rantai: **81** (`7ec0fa9..HEAD`, termasuk dua commit docs handoff).

- Spec: https://github.com/amaldevice/kanso-infer/issues/82
- Handoff rantai sebelumnya (konvensi linkage, pelajaran Fase 0–3): `docs/handoff-fase-0.md` §5
- ADR yang lahir dari rantai ini: `docs/adr/0002-fused-decode-scope.md`
- Keputusan workspace decode: `docs/decode-workspace.md`

## 2. Apa yang berubah di kode (peta cepat)

Modul yang **baru ada** karena rantai ini — baca ini sebelum menyentuh area terkait:

| Modul | Peran |
|---|---|
| `src/features.rs` | Registry fitur compile-checked: nama, status (`Wired`/`KernelOnly`/`Planned`), exec pointer, penanda `Option<fn()>`. Menghapus simbol produksi memecahkan build. Diaudit `tests/test_feature_registry.rs` |
| `src/model/paged_attn.rs` | Seam decode Metal: buffer pool + block table → `gpu_paged_attention_v2_buffer` |
| `src/model/fused_decode.rs` | Prologue fused (RMSNorm+QKV+RoPE) lewat `MetalWeights::decode_step`; tail per-op masih tensor path (lihat ADR-0002) |
| `src/kv_cache/kind.rs` | **Dua enum, jangan tertukar**: `KvCacheType` = `F32 \| Int8 \| Fp8` — sisi user, di-parse dari `--kv-type` (`FromStr`/`Display`), ada di `cli`/`chat`/`serve`. `KvStorageKind` = `F32(KvStorage) \| Quantized(QuantizedKvStorage)` — pool runtime yang dibangun engine. Metal menolak INT8/FP8 saat startup dengan `EngineError`, bukan panic |
| `src/bench/gate.rs` | Gate budget: baca `BenchmarkReport` JSON, exit 0/1/2, `skipped` ≠ `passed` |
| `src/bench/probes.rs` | 5 fungsi: `probe_gemv_peak_pct` (NEON, `None` off-aarch64), `probe_paged_attn_over_theoretical` (GPU, `None` tanpa Metal), `probe_sse_stream_consumed` (HTTP/1.1, bukan SDK), `run_probes`, `attention_geometry`. **Prinsip: menolak (`None`) saat kernel/hardware yang namanya disebut tidak ada.** RSS ada di `src/hardware.rs::get_peak_rss_mb`; perplexity di `src/bench/perplexity.rs` — bukan di sini |
| `src/bench.rs` (`KernelProbes`) | Empat field `Option<f64>`: `gemv_peak_pct`, `paged_attn_ctx4096_b1_over_theoretical`, `idle_gap_pct`, `sse_stream_consumed`. `None` → gate `skipped`, **tidak pernah** `0.0` yang terbaca sebagai pengukuran |
| `gates/*.json` | Budget F12/F15 + `no-measurement.json` + dua file self-check |
| `scripts/gate-ci.sh` | Runner gate di CI; tanpa GGUF → skipped, job tetap hijau |

Perubahan bentuk yang menyebar: PR #98 **52 file** (+4239/−301); seluruh rantai `7ec0fa9..HEAD`
**77 file** (+8672/−335). Inti sebarannya `Model::forward_batch*` dan tipe storage KV. Sudah
dipropagasi; hati-hati bila menambah parameter decode.

## 3. Status gate F12/F15 (inti sisa pekerjaan)

`gates/fase1.json` (6 baris) + `gates/fase2.json` (4 baris) → **9 dari 10 baris `measured`**, satu
sengaja tidak:

| Baris | Status | Catatan |
|---|---|---|
| `bandwidth_util_pct`, `batch_scale_ratio`, `spec_speedup_ratio` | measured | sudah ada sejak #84 |
| `gemv_peak_pct` | measured | kernel **NEON**; `None` di luar aarch64 |
| `paged_attn_ctx4096_b1_over_theoretical` | measured | kernel **GPU**; `None` tanpa Metal |
| `peak_rss_mb` | measured | `Option` — report legacy → skipped, bukan lulus |
| `perplexity_wikitext2` | measured | hormati `--eval-ppl <path>`; absen → skipped |
| `ctx_latency_ratio` | measured | pasangan konteks nyata, konteks besar sebagai pembilang |
| `sse_stream_consumed` | measured | pembacaan stream HTTP; **bukan** SDK |
| `idle_gap_pct` | **not-yet-instrumented** | butuh `MTLCommandBuffer.GPUStartTime/GPUEndTime` atau Metal System Trace; binding `metal` 0.29 tidak mengekspos |

## 4. Next step (berurutan)

### 4a. #99 — perbaiki CI merah (paling mendesak, murah)

`main` gagal di job `fmt + clippy + test` → `test_backend_enum` →
`class MTLResidencySetDescriptor could not be found` (objc2-metal, runner `macos-14`). Ini **bukan**
regresi rantai #82: **8 run `main` berturut-turut** gagal dengan panic identik, sejak `0ffd257`
(2026-10-17T17:50) — sebelum spec #82 ditulis.

Selama belum beres, **setiap PR akan terlihat merah** dan kegagalan nyata tercampur noise environment.
Dua job lain (`release-gates`, `budgets`) pass.

**Sumber panic sudah dilacak** (jangan tebak lagi): `src/hardware.rs:78` memanggil candle
`Device::new_metal(0)`, yang di dalamnya memakai `objc2-metal` dan `.expect()` pada
`MTLResidencySetDescriptor` → panic, bukan `Err`. Perhatikan `src/metal/context.rs:20` berbeda:
di sana `metal::Device::system_default()` mengembalikan `Option`, jadi **tidak** panic.

Arah perbaikan (tanpa dependensi baru): bungkus `Device::new_metal(0)` dengan
`std::panic::catch_unwind(AssertUnwindSafe(...))` di `Backend::device()` dan petakan panic menjadi
`EngineError::Metal("runner lacks MTLResidencySetDescriptor …")`. Alternatifnya menambah `objc2`
sebagai dependensi langsung untuk cek `Class::get`, tapi itu memperluas dep produksi hanya demi quirk
runner — kurang sepadan. Cek dulu `cargo test --test test_backend_enum` di mesin lokal: di macOS 27
class itu ada, jadi jalur panic tidak pernah tersentuh secara lokal dan harus diuji dengan
menginduksi panic.

### 4b. #101 — pengukuran nyata (`ready-for-human`)

Butuh: Mac dengan **Xcode penuh** (bukan CLT — `xctrace` menolak), file GGUF asli
(SmolLM2-360M Q8_0 + 135M), dan SDK SSE asli.

Jalankan: `cargo run --release --bin bench -- --gate gates/fase1.json …` lalu `gates/fase2.json`.
Instrumennya sudah siap; yang belum ada hanya angka + kalibrasi ambang. Detail di komentar #101.

Jangan turunkan ambang diam-diam — setiap pelonggaran harus tertulis.

### 4c. #82 — tutup setelah #101

### 4d. Opsional / kebersihan

- ~44 remote branch warisan sesi lama (`feat/*`, `spec/fase-1..3`, `ticket/42..62`), **semuanya sudah
  merged ke `main`**. Belum dihapus karena bukan scope rantai ini — menunggu keputusan pemilik repo.
- **2** `#[ignore]` tersisa (bukan 3 — kemunculan ketiga di `rg` adalah prosa di doc-comment):
  `tests/test_batched_forward.rs:267` dan `tests/test_speculative_draft_kv.rs:225`. Keduanya
  wall-clock gate `#[ignore = "release-only …"]` yang dijalankan job `release-gates`, bukan hijau palsu.

## 5. File yang perlu dieksplorasi

**Untuk #99 (CI):**
- `.github/workflows/ci.yml` — tiga job; yang gagal `check`
- `tests/test_backend_enum.rs:54` (`test_metal_device_is_honest_gpu`) dan `:72`
  (`test_metal_cpu_parity_mock`) — dua test yang panic; keduanya lewat `Backend::device()`
- `src/hardware.rs:78` — **titik panic**: candle `Device::new_metal(0)` → `objc2-metal` `.expect()`
- `src/metal/context.rs:20` — bandingkan: `metal::Device::system_default()` mengembalikan `Option`,
  jadi jalur ini aman dan tidak boleh ikut diubah
- `src/hardware.rs:102` — `device_label()`, dipakai assertion "Metal GPU"

**Untuk #101 (pengukuran):**
- `gates/fase1.json`, `gates/fase2.json` — budget dan `note` tiap baris
- `src/bench/gate.rs` — `GateRowStatus` (`Measured`/`NotYetInstrumented`), `GateVerdict`
  (`Passed`/`Violated`/`Skipped`), `metric_value` (~baris 387), `ratio_of` (~baris 648), exit 0/1/2
- `src/bench/probes.rs` — apa yang benar-benar diukur tiap probe
- `src/bench.rs` — `BenchmarkReport.probes: Option<KernelProbes>` (baris ~157)
- `src/bin/bench.rs` — flag `--gate`, `--gate-report`, `--eval-ppl`, `--ctx-lengths`, `--batch-sizes`,
  plus `--probes` dan `--sse-endpoint`
- `scripts/gate-ci.sh` — cara gate dipanggil

**Untuk memahami klaim README:**
- `src/features.rs` + tabel Feature Status di `README.md` — satu sumber kebenaran, diaudit test
- `docs/adr/0002-fused-decode-scope.md` — kenapa satu command-buffer per step **belum** tercapai

**Untuk menyentuh jalur decode:**
- `src/model.rs` — `forward_decode_step` (:1078), `forward_batch` (:1331), `metal_paged_attention` (:852)
- `src/model/paged_attn.rs`, `src/model/fused_decode.rs`, `src/kv_cache/kind.rs`
- `src/engine.rs` — `new_with_budget_and_kv` (:69), `step` (:211), `speculative_step_sampled` (:613)

**Untuk memverifikasi handoff ini sendiri:** `.codegraph/` sudah terindeks (7.5 MB) — pakai
`codegraph_explore` lewat codemode, lebih akurat dan jauh lebih murah daripada grep+read berulang.

## 6. Pelajaran operasional (mahal, jangan diulang)

- **4 agent paralel di mesin 6 core / 2 performance core membuat laptop freeze** (load average 142).
  Jalankan implementer **sekuensial**: satu agent, satu tiket, satu file test per verifikasi.
- **Sesi mati 4× karena prompt >450k token** (output `cargo test` penuh, timeout 900s, diff besar).
  Wajib: `cargo test --test <file>` (jangan pernah `cargo test` polos), `| tail -20` di setiap
  perintah, `timeout` ≤300s, `git diff --stat`.
- **Commit per milestone** (RED lalu GREEN) menyelamatkan kerja saat stream error. Sesi tanpa commit
  kehilangan ~1 jam analisis.
- **Spec bisa salah.** Spec #82 mencantumkan D17 (`workspace.clear()` per decode) sebagai debt padahal
  sudah dihapus di Fase-2 F12. Koreksi dicatat sebagai komentar di #82, bukan suntingan badan spec.
- **Code review dua-axis menemukan 7 klaim hollow yang lolos dari AC tiketnya sendiri** (lihat badan
  PR #102). Jalankan review meski semua AC tercentang.

## 7. Konvensi yang masih berlaku

Dari `docs/handoff-fase-0.md` §5 — tidak berubah:

- Commit memuat `(ticket #N)`. **Jangan** pakai `fix #N` / `closes #N` (pernah menutup issue prematur).
- Laporan implementer: `TRACK: ticket #N | branch … | integration … | PR draft #M | spec #S | unblocks #X`
  plus `BASE / RED / GREEN / SCOPE / DEBT`.
- Satu tiket = satu worktree + satu branch; subagent **tidak** merge ke integration, tidak menutup
  issue, tidak menghapus worktree.
- Tiap perpindahan label fitur (✅/🧪/📝) wajib lewat `src/features.rs`; audit akan gagal bila README
  dan registry tidak sinkron.

## Suggested skills

- **`implement-spec`** — orkestrasi tiket: worktree per tiket, merger, frontier. Pakai ini untuk #99.
- **`tdd`** — wajib untuk tiap implementer subagent (red-green-refactor).
- **`code-review`** — review dua-axis (Standards + Spec) sebelum merge. Di rantai ini ia menemukan 7
  klaim hollow yang lolos dari AC; jangan dilewati.
- **`herdr`** — delegasi ke subagent (`herdr pane split` + `herdr agent start --kind pi`), plus polling
  `herdr agent list` / `herdr agent read`. Ingat batas konkurensi di §6.
- **`pr`** — menulis badan PR (Summary / Evidence / Merge Danger).
- **`handoff`** — tulis handoff baru bila sesi terputus lagi.
- **`wizard`** — bila #101 dijalankan: setup Xcode penuh + unduh GGUF adalah langkah manusia.
