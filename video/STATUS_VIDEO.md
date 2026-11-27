# NEBians News Ep.1 (v2 Remake) — Friday Weekly Recap Status

Goal: **5+ minute (342.0 s / 5m 42s)** Material 3 Expressive motion-graphics weekly news episode, narrated by **Sakhiya-One v2 Studio** (`docs/models_sakhiya_one_vocos/model.int8.onnx`, 23.011 MB INT8 ONNX, 24 kHz Vocos, `news` style), rendered at 1080p30 on MoLab (`sb-e42961993844f617.sb.molab.run`), and published to GitHub Releases (`news-ep1`).

## Key Updates in the v2 Remake
1. **Friday Weekly Recap Format & Strict Date Window (`Sat 21 Nov – Fri 27 Nov 2026`)**:
   - Episodes are now published **every Friday** as a full weekly recap.
   - Strictly covers verified stories from **Saturday, 21 Nov 2026 to Friday, 27 Nov 2026** (zero stories older than Nov 21).
   - **Removed all NEBians Android app segments** (`app` scene replaced by a dedicated **Nepal Education Sector Weekly Desk** `edu`).
   - Covers major **Global AI, Space/Cloud Hardware, Semiconductors, Cybersecurity/Linux, Autonomous Mobility/Consumer Devices**, and **Nepal Education Sector** updates with full multi-sentence context per story (`46` lines across `11` scenes, `342.0 s` / `5m 42s`).
2. **Updated Sakhiya-One v2 Intro & Native Nepali Script Purity**:
   - Introduces **Sakhiya-One v2** (`सखिया वान को नयाँ दोस्रो संस्करण`) as our current-phase neural text-to-speech model with simple, accessible tech details (`23 MB` Studio edition, `8.5 MB` Compact edition, speaks Nepali and Sanskrit, runs directly in the browser at `24 kHz` studio quality over `30×` faster than real-time, cleaner and clearer than Phase 1 while continuing to improve every week).
   - 100% pure native Devanagari script using simple, natural Nepali vocabulary.
   - Zero `चलो` / `चलौँ` (uses `अब लागौँ आजको पहिलो समाचारतर्फ।`).
3. **Complete Material 3 Expressive (M3X) Visual & Motion Overhaul (`video/web/`)**:
   - **5 Expressive Tonal Palettes**: `blue` (Royal Cobalt), `light` (Alabaster & Periwinkle), `dark` (Obsidian Slate), `teal` (Emerald Cyber/Space), and `plum` (Editorial Indigo for the Nepal Education Desk).
   - **12 Morphable 180-Vertex M3 Expressive Silhouettes**: `circle`, `cookie`, `cookie12`, `sunny`, `flower`, `clover`, `burst`, `squircle`, `pebble`, `gem`, `hex`, `shieldShape` with coupled spring physics (`spring`, `springSoft`, `eoback`).
   - **Broadcast Header & 11-Chapter Progress Strip**: Live section badge, 11-scene interactive progress bar, and real-time voice-reactive waveform telemetry pill.
   - **Compact, Non-Cluttering Bilingual Subtitle Pill (`caption()` in `ui.js`)**: Sleek `62px`-tall floating pill at the bottom edge with `22px` Devanagari word-by-word karaoke highlight and `15px` English subtitle, leaving ~90% of the 1920×1080 canvas completely unobstructed.

## Weekly Story Lineup (`video/script.json` · 21–27 Nov 2026)
- `hello` + `about` + `series`: Meet Sakhiya-One v2 + Friday Weekly Recap format (`21–27 Nov 2026` 7-day calendar strip).
- `n1` (**Space & Cloud** · 25–26 Nov): FCC approves SpaceX **Starmind** constellation of up to **1,000,000** solar-powered orbital AI data center satellites linked via lasers to Starlink.
- `n2` (**Chips & Semiconductors** · 23–27 Nov): Nvidia board reinstates **Jensen Huang** as CEO (+30% Monday surge, -6% Tuesday dip after no-succession-plan remarks), halts AI server shipments to Malaysia, Singapore, Thailand & Indonesia under US export curbs, while **Huawei** teases a next-gen HBM memory chip.
- `n3` (**AI Models & Software Engineering** · 23–26 Nov): OpenAI brings **GPT-6.2 Sol** to the main ChatGPT tab, tests encrypted **Private Inference**, and demonstrates protein design; Anthropic launches **Claude Ship** PR agent and migrates its API from Python to **Rust** (-70% CPU usage) while surpassing 700M monthly MCP downloads.
- `n4` (**Cybersecurity, Linux & Policy** · 23–26 Nov): OpenAI & Cisco launch **Shirudo** to protect live Linux servers (plus AWS Nitro/Firecracker isolation); 3.5B records exposed & US FinCEN offline after cyber intrusion; **New York** signs a 5-year ban on AI chatbot toys ($15,000/day fine); 42 US states move to block Character.AI's 21M private chat auction.
- `n5` (**Mobility & Consumer Tech** · 22–26 Nov): **Tesla FSD v15** rollout & **Uber-Lucid-Nuro** Bay Area robotaxi testing; **Apple iOS 27.3** & **Nothing Ear Note** recording earbuds; **Mozilla Firefox 161** per-site auto VPN; **Walmart** & **Amazon** talks on Meta's **Muse** shopping agent.
- `edu` (**Nepal Education Sector Desk** · 21–26 Nov): **Tribhuvan University (TU)** directs affiliated colleges to submit mandatory self-declarations on labs and libraries ahead of semester reforms; **Medical Education Commission (MEC)** publishes bachelor entrance (`MECEE-BL`) results & Govt holds medical education reform talks; **Kathmandu Metropolitan City (KMC)** lays foundation stone for the **Autism School & Research Centre** in Anamnagar; **Sudurpaschim University** National Hackathon, **Kathmandu University (KU)** Geomatics entrance schedule & Ministry R&D funding priority.
- `fast` (**Quick Weekly Tech Roundup** · 23–27 Nov): **PrismML Ternary Bonsai 27B** (`54 GB -> 5.9 GB`), **Magnitude** SSD model streaming (`66 GB` on `24 GB` GPU), **Tencent DFly2** (`+81%` speed) & Canada `$1B` AI compute fund, and **Tidal** removing fully AI-generated music starting Dec 9.
- `outro`: Weekly sign-off (`धन्यवाद र नमस्ते · अर्को शुक्रबार फेरि भेटौँला`).
