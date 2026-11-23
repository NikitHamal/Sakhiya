// sakhiya-cli: Standalone C++17 TTS CLI linked against ggml
#include "sakhiya_ggml.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -m, --model <path.gguf>      Path to Sakhiya GGUF model file\n"
              << "  -t, --text <utf8_text>       Input text (Nepali / Sanskrit / Roman / Mixed)\n"
              << "  -l, --lang <ne|sa|auto|...>  Language code (default: ne)\n"
              << "  -s, --style <style_name>     Style: neutral|news|conversational|storytelling|warm_soft|slow_clear|expressive|vedic_chant\n"
              << "  -o, --out <output.wav>       Output 24 kHz 16-bit PCM WAV path (default: sakhiya_cpp.wav)\n"
              << "  -j, --threads <N>            CPU threads (default: 4)\n"
              << "  --rate <float>               Speaking speed multiplier (0.5 - 2.0)\n"
              << "  --pitch-shift <float>        Pitch shift in semitones/octaves (-0.6 - 0.6)\n"
              << "  --pitch-scale <float>        Expressiveness / tone scale (0.5 - 1.8)\n"
              << "  --pause-scale <float>        Pause duration multiplier (0.3 - 2.5)\n"
              << "  --energy-scale <float>       Projection / warmth multiplier (0.6 - 1.6)\n"
              << "  --svara <float>              Vedic Svara depth multiplier (default: 1.0)\n"
              << "  --batch <manifest.tsv>       Batch synthesize tab-separated lines: id\\tlang\\tstyle\\tout_wav\\ttext\n"
              << "  --dump-stages <dir>          Dump exact intermediate tensors (ids, x_enc, logdur, dur, mel, spec, wav) for diff gate\n"
              << "  --g2p-only                   Print JSON with phones, ids, pitch_add, dm, and exit\n"
              << "  --slp1-only                  Print JSON with Sanskrit phonemize_slp1 output and exit\n"
              << "  --laya-only                  Print JSON with Sanskrit laya_scales(v2=True) output and exit\n";
}

static std::string escape_json(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path = "/work/tts/models/sakhiya_f32.gguf";
    std::string text = "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।";
    std::string lang = "ne";
    std::string style = "news";
    std::string out_wav = "sakhiya_cpp.wav";
    std::string dump_dir;
    std::string batch_file;
    int threads = 4;
    float svara = 1.0f;
    bool has_override = false;
    float ov_rate = -1.0f, ov_ps = -999.0f, ov_psc = -1.0f, ov_pause = -1.0f, ov_eng = -1.0f;
    bool g2p_only = false;
    bool slp1_only = false;
    bool laya_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) model_path = argv[++i];
        else if ((a == "-t" || a == "--text") && i + 1 < argc) text = argv[++i];
        else if ((a == "-l" || a == "--lang") && i + 1 < argc) lang = argv[++i];
        else if ((a == "-s" || a == "--style") && i + 1 < argc) style = argv[++i];
        else if ((a == "-o" || a == "--out") && i + 1 < argc) out_wav = argv[++i];
        else if ((a == "-j" || a == "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--rate" && i + 1 < argc) { ov_rate = std::atof(argv[++i]); has_override = true; }
        else if (a == "--pitch-shift" && i + 1 < argc) { ov_ps = std::atof(argv[++i]); has_override = true; }
        else if (a == "--pitch-scale" && i + 1 < argc) { ov_psc = std::atof(argv[++i]); has_override = true; }
        else if (a == "--pause-scale" && i + 1 < argc) { ov_pause = std::atof(argv[++i]); has_override = true; }
        else if (a == "--energy-scale" && i + 1 < argc) { ov_eng = std::atof(argv[++i]); has_override = true; }
        else if (a == "--svara" && i + 1 < argc) { svara = std::atof(argv[++i]); }
        else if (a == "--batch" && i + 1 < argc) batch_file = argv[++i];
        else if (a == "--dump-stages" && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--g2p-only") g2p_only = true;
        else if (a == "--slp1-only") slp1_only = true;
        else if (a == "--laya-only") laya_only = true;
        else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (slp1_only) {
        auto r = sakhiya::phonemize_slp1(text);
        std::cout << "{\"ids\":[";
        for (size_t i = 0; i < r.ids.size(); ++i) std::cout << (i ? "," : "") << r.ids[i];
        std::cout << "],\"weight\":[";
        for (size_t i = 0; i < r.weight.size(); ++i) std::cout << (i ? "," : "") << r.weight[i];
        std::cout << "],\"off\":[";
        for (size_t i = 0; i < r.off.size(); ++i) std::cout << (i ? "," : "") << std::setprecision(8) << r.off[i];
        std::cout << "],\"syms\":[";
        for (size_t i = 0; i < r.syms.size(); ++i) std::cout << (i ? "," : "") << "\"" << escape_json(r.syms[i]) << "\"";
        std::cout << "]}\n";
        return 0;
    }

    if (laya_only) {
        auto r = sakhiya::laya_scales(text, true, true);
        std::cout << "{\"ph\":[";
        for (size_t i = 0; i < r.ph.size(); ++i) std::cout << (i ? "," : "") << "\"" << escape_json(r.ph[i]) << "\"";
        std::cout << "],\"off\":[";
        for (size_t i = 0; i < r.off.size(); ++i) std::cout << (i ? "," : "") << std::setprecision(8) << r.off[i];
        std::cout << "],\"dm\":[";
        for (size_t i = 0; i < r.dm.size(); ++i) std::cout << (i ? "," : "") << std::setprecision(8) << r.dm[i];
        std::cout << "]}\n";
        return 0;
    }

    if (g2p_only) {
        std::vector<std::string> ph;
        std::vector<double> off;
        std::vector<double> dm;
        bool has_vm = false;
        if (lang == "sa") {
            auto ch = sakhiya::compile_chandas(text, 75.0, 215.0, true);
            ph = ch.ph; off = ch.off; dm = ch.dm; has_vm = ch.has_vedic_marks;
        } else {
            auto acc = sakhiya::text_to_phones_acc(text, lang, true);
            ph = acc.first; off = acc.second; dm = sakhiya::phone_dm(ph);
        }
        const auto & sym2id = sakhiya::get_sym2id();
        std::vector<int> ids;
        std::vector<float> pitch_add, dm_arr;
        float sv_sc = (lang == "sa") ? (has_vm ? 0.52f : 0.48f) : 1.15f;
        for (size_t i = 0; i < ph.size(); ++i) {
            auto it = sym2id.find(ph[i]);
            if (it != sym2id.end() && it->second >= 0) {
                ids.push_back(it->second);
                pitch_add.push_back(static_cast<float>(off[i]) * sv_sc);
                dm_arr.push_back(static_cast<float>(dm[i]));
            }
        }
        std::cout << "{\"ph\":[";
        for (size_t i = 0; i < ph.size(); ++i) std::cout << (i ? "," : "") << "\"" << escape_json(ph[i]) << "\"";
        std::cout << "],\"ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
        std::cout << "],\"pitch_add\":[";
        for (size_t i = 0; i < pitch_add.size(); ++i) std::cout << (i ? "," : "") << std::setprecision(8) << pitch_add[i];
        std::cout << "],\"dm\":[";
        for (size_t i = 0; i < dm_arr.size(); ++i) std::cout << (i ? "," : "") << std::setprecision(8) << dm_arr[i];
        std::cout << "]}\n";
        return 0;
    }

    try {
        sakhiya::SakhiyaGGML engine;
        auto t_load0 = std::chrono::high_resolution_clock::now();
        engine.load_model(model_path, threads);
        auto t_load1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

        if (!batch_file.empty()) {
            std::ifstream bf(batch_file);
            if (!bf) throw std::runtime_error("Cannot open batch file: " + batch_file);
            std::string line;
            while (std::getline(bf, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                std::string cid, b_lang, b_style, b_out, b_text;
                if (!std::getline(iss, cid, '\t')) continue;
                if (!std::getline(iss, b_lang, '\t')) continue;
                if (!std::getline(iss, b_style, '\t')) continue;
                if (!std::getline(iss, b_out, '\t')) continue;
                if (!std::getline(iss, b_text)) continue;
                auto tb0 = std::chrono::high_resolution_clock::now();
                std::vector<float> b_wav = engine.say(b_text, b_lang, b_style, svara, nullptr, nullptr);
                auto tb1 = std::chrono::high_resolution_clock::now();
                double b_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
                double b_sec = static_cast<double>(b_wav.size()) / 24000.0;
                double b_rtf = (b_ms / 1000.0) / std::max(b_sec, 1e-6);
                sakhiya::write_wav_pcm16(b_out, b_wav, 24000);
                std::cout << "[sakhiya-cli batch] id=" << cid << " out=" << b_out
                          << " dur=" << std::fixed << std::setprecision(2) << b_sec << "s"
                          << " synth=" << std::setprecision(1) << b_ms << "ms"
                          << " RTF=" << std::setprecision(4) << b_rtf << "\n";
            }
            return 0;
        }

        sakhiya::StyleConfig ctl = sakhiya::get_style_config(style);
        if (lang == "sa" && style != "chant" && style != "vedic_chant" && style != "storytelling") {
            ctl.rate *= 0.96f;
            ctl.pause_scale *= 1.05f;
            ctl.pitch_scale = 1.00f;
        }
        if (ov_rate > 0.0f) ctl.rate = ov_rate;
        if (ov_ps > -900.0f) ctl.pitch_shift = ov_ps;
        if (ov_psc > 0.0f) ctl.pitch_scale = ov_psc;
        if (ov_pause > 0.0f) ctl.pause_scale = ov_pause;
        if (ov_eng > 0.0f) ctl.energy_scale = ov_eng;
        const sakhiya::StyleConfig * ctl_ptr = has_override ? &ctl : nullptr;

        sakhiya::StageDump dump;
        sakhiya::StageDump * dump_ptr = dump_dir.empty() ? nullptr : &dump;

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> wav = engine.say(text, lang, style, svara, ctl_ptr, dump_ptr);
        auto t1 = std::chrono::high_resolution_clock::now();

        double synth_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double audio_sec = static_cast<double>(wav.size()) / 24000.0;
        double rtf = (synth_ms / 1000.0) / std::max(audio_sec, 1e-6);

        sakhiya::write_wav_pcm16(out_wav, wav, 24000);
        if (dump_ptr) {
            std::string cmd = "mkdir -p \"" + dump_dir + "\"";
            int rc = std::system(cmd.c_str());
            (void)rc;
            sakhiya::write_raw_bin(dump_dir + "/ids.i64.bin", dump.ids.data(), dump.ids.size() * sizeof(int64_t));
            sakhiya::write_raw_bin(dump_dir + "/pitch_add.f32.bin", dump.pitch_add.data(), dump.pitch_add.size() * sizeof(float));
            sakhiya::write_raw_bin(dump_dir + "/dm.f32.bin", dump.dm.data(), dump.dm.size() * sizeof(float));
            sakhiya::write_raw_bin(dump_dir + "/x_enc.f32.bin", dump.x_enc.data(), dump.x_enc.size() * sizeof(float));
            sakhiya::write_raw_bin(dump_dir + "/logdur.f32.bin", dump.logdur.data(), dump.logdur.size() * sizeof(float));
            sakhiya::write_raw_bin(dump_dir + "/dur.i32.bin", dump.dur.data(), dump.dur.size() * sizeof(int32_t));
            sakhiya::write_raw_bin(dump_dir + "/mel.f32.bin", dump.mel.data(), dump.mel.size() * sizeof(float));
            sakhiya::write_raw_bin(dump_dir + "/spec_re.f32.bin", dump.spec_re.data(), dump.spec_re.size() * sizeof(float));
            sakhiya::write_raw_bin(dump_dir + "/spec_im.f32.bin", dump.spec_im.data(), dump.spec_im.size() * sizeof(float));
            sakhiya::write_raw_bin(dump_dir + "/wav.f32.bin", dump.wav.data(), dump.wav.size() * sizeof(float));

            std::ofstream meta(dump_dir + "/meta.json");
            meta << "{\n"
                 << "  \"L\": " << dump.ids.size() << ",\n"
                 << "  \"T_mel\": " << dump.T_mel << ",\n"
                 << "  \"T_spec\": " << dump.T_spec << ",\n"
                 << "  \"wav_samples\": " << dump.wav.size() << ",\n"
                 << "  \"sample_rate\": 24000,\n"
                 << "  \"load_ms\": " << load_ms << ",\n"
                 << "  \"synth_ms\": " << synth_ms << ",\n"
                 << "  \"audio_sec\": " << audio_sec << ",\n"
                 << "  \"rtf\": " << rtf << "\n"
                 << "}\n";
        }

        std::cout << "[sakhiya-cli] out=" << out_wav
                  << " samples=" << wav.size()
                  << " dur=" << std::fixed << std::setprecision(2) << audio_sec << "s"
                  << " synth=" << std::setprecision(1) << synth_ms << "ms"
                  << " RTF=" << std::setprecision(4) << rtf
                  << " (threads=" << threads << ")\n";
    } catch (const std::exception & e) {
        std::cerr << "[sakhiya-cli] ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
