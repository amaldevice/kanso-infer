# Handoff — Fase-0 Correctness (spec #59)

> Untuk agent penerus: lanjutkan implementasi Fase-0 dari frontier `#44`.
> Dokumen ini ringkasan state, bukan duplikasi spec — baca artifact yang dirujuk.

## 0. Flow end-to-end: Fase → Spec → Tiket → PR

> P0/P1/P2 = severity tiap bug. Fase 0–3 = urutan pengerjaan dari tracking #58.
> Satu Fase berisi beberapa level P — dikelompokkan berdasar ketergantungan kerja, bukan severity.
> Prinsip #58 yg mengikat semua: angka performa tertelusur ke trace; fitur README tertelusur
> ke jalur `kanso serve`; kalau tidak, itu rencana, bukan fitur.

```text
#58 (tracking: correctness → GPU → serving → honesty)
 └── #59 Fase-0 ──▶ #60 Fase-1 ──▶ #61 Fase-2 ──▶ #62 Fase-3   (native deps, terverifikasi)
      ├── #57 ✅ (PR #63 merged)   [P2: CI + mock random + golden — fondasi]
      ├── #43 ✅kode (PR #64 draft)[P0: chunked-prefill amnesia — mask offset + gather]
      ├── #44 🟡 FRONTIER          [P0: prefix use-after-free — invalidate + LRU]
      └── #56 ⬜ blocked-by #44    [P2: OOM/preemption/EOS/leak]
```

- **Fase 0 (spec #59): jangan optimasi output yg salah.** Urutan: #57 dulu (tanpa mock
  random bug #43 tak terlihat — terbukti golden gagal diff 11.67), lalu #43, #44 (aman karena
  prefill sudah benar), #56 (preemption aman karena invalidate benar). Keluar: review dua-axis
  → merge PR #64 → close #59.
- **Fase 1 (spec #60, blocked-by #59): GPU beneran.** #42 backend enum → #47 quantized/F16
  → #49 1 command-buffer + weight resident + metallib AOT → #50 GEMV multi-threadgroup +
  #51 paged-attn partisi+reduce; #52 AMX independen. Gate: ukur GPU di atas output salah = sia-sia.
- **Fase 2 (spec #61, blocked-by #60): throughput serving.** #45 batched forward (butuh #51 +
  Fase-0 hijau) ↔ #48 tanpa gather-copy; #46 server SSE + engine loop (butuh #56 + #45);
  #55 draft-KV (butuh #48). Gate: satu forward per step.
- **Fase 3 (spec #62, blocked-by #61): kejujuran.** #53 telemetri dari model+KV aktual
  (butuh #47 + #45); #54 tabel README wired/kernel-only/planned (jujur setelah Fase 0–2
  membuktikan wiring nyata). Gate: tiap klaim tertelusur atau berlabel planned.

Eksekusi per tiket mengikuti skill `implement-spec`: satu tiket = satu worktree + satu branch
(`ticket/<N>-<slug>`) berbasis integration branch (`spec/fase-<k>-...`); subagent TDD
(red-green-refactor); merger gabung ke integration; frontier maju; semua tiket done →
`code-review` → fix → merge PR → resolve tiket + spec → cleanup. Detail pola di §5.

## 1. Status sekarang (2026-10-16)

| Tiket | State | Catatan |
|---|---|---|
| #57 CI + mock random | CLOSED | via PR #63 (MERGED ke `main`) |
| #43 chunked-prefill parity | OPEN, kode DONE | fix merged ke `spec/fase-0-correctness`; **tinggal close formal** (lihat §4) |
| #44 prefix invalidate | OPEN, FRONTIER | worktree siap, belum ada kode |
| #56 engine robustness | OPEN | blocked by #44 (native dep) |
| #59 spec Fase-0 | OPEN | close setelah semua tiket done + review final |
| #58 tracking + #60/#61/#62 specs Fase-1–3 | OPEN | rantai dep: #59 → #60 → #61 → #62 |
| PR #64 (draft) | OPEN | integration branch Fase-0, body ada tabel Tracking |

Branch: `main` @ `aeffd8c` (isi PR #63). Integration: `spec/fase-0-correctness` @ `71b6635` (= main + fix #43).
Worktree `#44` ada dan bersih: `/Users/amalanfadil/.herdr/worktrees/from_scratch/ticket-44-prefix-invalidate` (branch `ticket/44-prefix-invalidate`, base 71b6635, tanpa file baru).
Pane/agent lama (`impl44`, `impl44b`, reviewer) sudah tidak ada — buat baru saat lanjut.

## 2. Task graph Fase-0

```
#57 ✅ → #43 ✅(kode) → #44 🟡 FRONTIER → #56 ⬜ → review → merge PR #64 → close #59
```

Native GitHub dependencies sudah dipasang; peta lintas-fase dikomentar di #58.

## 3. Context pointers (baca ini, jangan tebak)

- Spec: #59 (problem/solution/testing decisions) — https://github.com/amaldevice/kanso-infer/issues/59
- Tiket aktif: #44 — https://github.com/amaldevice/kanso-infer/issues/44 (acceptance: skenario A→B→C + LRU evict)
- Tracking PR: #64 (tabel Tracking di body) — https://github.com/amaldevice/kanso-infer/pull/64
- Eksplorasi codebase: `/tmp/explorer_report.md` (jalur produksi + modul tak-tersambung)
- Review sebelumnya: `/tmp/review_standards.md`, `/tmp/review_spec.md` (pola review dua-axis)

## 4. Langkah berikutnya (berurutan)

1. **Close #43 formal** — fix sudah di integration branch (commit `71b6635`, komentar sudah dipost di #43). Tutup dengan komentar: verified via `test_chunked_golden` hijau chunk 4/7/16. (Opsional: tunggu sampai PR #64 merge; pola #57 adalah close-setelah-merge.)
2. **Implement #44** — TDD di worktree di atas: RED `tests/test_prefix_invalidate.rs` (A→B→C + LRU), GREEN minimal (invalidate-on-reuse + batas LRU wajib; hash berantai O(L) disarankan, catat utang bila dilewat; claim O(1)/CoW opsional). Batas scope: JANGAN sentuh engine scheduling/EOS (#56), backend/GPU, forward_prefill logic.
3. **Implement #56** (setelah #44) — scheduling + preemption + EOS/free + drain finished.
4. **Code-review dua-axis** di `spec/fase-0-correctness` vs `main`, fix temuan, merge PR #64 (tulis body pakai skill `pr`), close #44/#56/#59, hapus branch + worktree.

## 5. Konvensi linkage (wajib tiap implementasi)

- Commit message selalu memuat `(ticket #N)`.
- Laporan implementer selalu berformat:
  `TRACK: ticket #N | branch … | integration … | PR draft #M | spec #S | unblocks #X` + `BASE / RED / GREEN / SCOPE / DEBT`.
- Komentar issue selalu menyebut branch + PR. Update tabel Tracking di body PR #64 tiap tiket masuk.
- Aturan subagent (skill `implement-spec`): satu tiket = satu worktree + satu branch; verifikasi base sebelum mulai; merge ujung integration branch sebelum lapor; TIDAK merge integration / close issue / cleanup dari subagent.

## 6. Pelajaran sesi ini

- Mock `ones` membuat suite buta — semua test numerik baru wajib mock random seeded (sudah di `main`).
- `test_chunked_golden` pola bukti-gigi: RED pra-fix (diff 11.67) → hijau pasca-fix; pakai pola sama untuk #44.
- 5 test speculative di-`#[ignore]` (tracking komentar di #43) + 1 chunked-golden ignore dihapus pasca-#43; 5 speculative masih ignore karena sebab lain — verifikasi ulang pasca-#44/#56, re-enable bila hijau.
- Error Cloudflare 524 dari provider model pernah menggugurkan subagent 2x — mitigasi: instruksikan subagent commit lokal per milestone (RED dulu, GREEN kemudian) agar retry tidak mengulang dari nol.
- #43 pernah tertutup prematur saat merge PR (auto-close dari commit message) padahal fix belum di `main` — reopen + jelaskan; hati-hati kata kunci `fix #N` di commit sebelum merge ke `main`.

## Suggested skills

- `implement-spec` — orkestrasi tiap tiket (worktree + merger + frontier).
- `tdd` — instruksi wajib tiap implementer subagent (red-green-refactor).
- `code-review` — review dua-axis (Standards + Spec) sebelum merge PR #64.
- `pr` — tulis body PR #64 saat ready (Summary / Evidence / Merge Danger + tabel Tracking).
- `handoff` — tulis handoff baru bila sesi terputus lagi.

## 7. Penutup Fase-0 — DONE 2026-10-17 (STOP di sini)

> Fase-0 CLOSED. PR #64 MERGED ke `main` (`7f13ec8`). Spec #59 + tiket #43/#44/#56 CLOSED.
> Instruksi user: stop di Fase-0, jangan mulai Fase-1 (#60).

| Tiket | PR tiket | State |
|---|---|---|
| #57 CI + mock random | #63 (merged duluan) | CLOSED |
| #43 chunked-prefill parity | — (langsung di integration) | CLOSED |
| #44 prefix invalidate | #65 | CLOSED |
| #56 engine robustness | #66 | CLOSED |
| #59 spec Fase-0 | #64 (integration, merged) | CLOSED |

- **#44**: `block_to_keys` reverse index + `invalidate_block` di 1 situs alokasi
  (`ensure_blocks` pasca-#56) + LRU bound 64 + touch-on-hit; chained-hash O(L)
  dilewat eksplisit (whole-prefix-key lookup sudah O(1)/level).
- **#56**: `ensure_blocks()` + preempt-newest (recompute, fold-back emitted) +
  EOS retire same-step + `drain_finished()` (server/CLI/chat); `SamplingParams`
  di-sequence, terplumb server + CLI (chat REPL + server-EOS = tiket sendiri).
- **Review dua-axis** (`/tmp/review_fase0_final.md`): 0 blocker, #44/#56 PASS,
  fix F1–F6 applied (F3: verifier invalidate + speculative free; F4/F5 decision-variant).
- **Suite di main**: 123 passed, 5 ignored (36 suites; 5 speculative ignore pra-ada).
- Worktree/branch tiket dihapus; `main` @ `7f13ec8` bersih.
- Lanjut (bila diminta): #60 Fase-1 GPU — mulai dari frontier #42, pola sama.

## 8. Penutup Fase-1 — DONE 2026-10-17

> Fase-1 CLOSED. PR #68 MERGED ke `main` (`8ea4fb8`). Spec #60 + tiket
> #42/#47/#49/#50/#51/#52 CLOSED (auto-close pas merge).

| Tiket | PR tiket | State |
|---|---|---|
| #42 backend enum | #71 | CLOSED |
| #51 paged-attn v2 | #67 | CLOSED |
| #50 GEMV multigroup | #69 | CLOSED |
| #52 AMX no-memcpy | #70 | CLOSED |
| #49 cmd-buffer + AOT | #72 | CLOSED |
| #47 quantized weights | #73 | CLOSED |
| #60 spec Fase-1 | #68 (integration, merged) | CLOSED |

- **#42**: `Backend` enum + `Device::new_metal(0)` + honest label; foo = error.
- **#51**: split-K partitions + merge kernel + GQA sharing; parity 8/8.
- **#50**: multi-TG coalesced float4 + simd_sum; parity 960/2048/4096.
- **#52**: CblasTrans zero-copy; 4KB/step (was 7.4MB); 1.48x mock-dims.
- **#49**: `MetalWeights` resident + 1 cmdbuf/step + build.rs metallib AOT
  (strict-proof di Xcode CI; box ini CLT-only).
- **#47**: `LinearWeight(QMatMul)` resident; RSS 493MB; 3.89x; PPL d0.18%
  (synthetic-proxy; real-GGUF hook `KANSO_GGUF=`).
- **Review dua-axis** (`/tmp/review_fase1_final.md`): #42/#51 PASS;
  #47/#49/#50/#52 struktur-parity hijau, angka-bandwidth manual. Fix F1–F11
  applied (F1: decode_step harness-only documented — full Metal decode loop =
  tiket sendiri; F12 pengukuran manual = debt).
- **Suite di main**: 155 passed, 5 ignored (41 suites).
- **Debt pengukuran F12** (butuh target Mac + Instruments + GGUF asli):
  GEMV≥60%, attn≤2x teoritis, idle-gap<5% wired-step, RSS/tok/s/PPL real-file.
- Worktree/branch tiket dihapus; `main` @ `8ea4fb8` bersih.
- Lanjut: #61 Fase-2 serving — frontier #45/#48 (butuh #51 + Fase-0 hijau).

## 9. Penutup Fase-2 — DONE 2026-10-18

> Fase-2 CLOSED. PR #75 MERGED ke `main` (`eae6e1f`). Spec #61 + tiket
> #45/#48/#46/#55 CLOSED (auto-close pas merge).

| Tiket | PR tiket | State |
|---|---|---|
| #48 paged decode | #74 | CLOSED |
| #45 batched forward | #76 | CLOSED |
| #55 draft-KV | #77 | CLOSED |
| #46 server SSE | #78 | CLOSED |
| #61 spec Fase-2 | #75 (integration, merged) | CLOSED |

- **#48**: `paged_kv_tokens` zero-copy view + strided AMX + rope table;
  cpu 1.262x, metal 1.299x, amx 1.063x; 353MB→18MB/step.
- **#45**: `forward_batch` 1 GEMM/step + `write_kv_slots`; batch-8 4.2x cpu /
  5x amx (release, mock shapes).
- **#55**: draft-KV + Leviathan + verify-RoPE fix (root cause 5 ignored tests
  sejak #43 — kini re-enabled); proxy 2.4x; real-pair 0.84x documented.
- **#46**: engine-loop thread + per-request mpsc + real tokenizer + SSE
  `chat.completion.chunk`; 8-konkuren hijau.
- **Review dua-axis** (`/tmp/review_fase2_final.md`): 4 CONDITIONAL PASS,
  fix F1–F14 applied (F3 claim scoped non-spec; F5/F7 ignore + release-job).
- **Suite**: debug hijau kecuali decode-latency (environmental: lolos isolasi
  base+tip, gagal hanya full-paralel).
- **Debt F15** (target Mac + GGUF + SDK): batch-8 real, ctx-latency real,
  spec real-pair speedup, SDK SSE run.
- Worktree/branch tiket dihapus; `main` @ `eae6e1f` bersih.
- Lanjut: #62 Fase-3 honesty — #53 telemetri + #54 README table.
