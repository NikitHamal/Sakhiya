#pragma once
// Sakhiya C++17 Front-End: 100% exact parity with sakhiya/{symbols,numbers,translit,g2p,sanskrit,chandas,mixed,onnx_tts}.py
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sakhiya {

// ---------------- UTF-8 <-> UTF-32 helpers ----------------
inline std::u32string utf8_to_u32(const std::string & s) {
    std::u32string out;
    out.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        if (c < 0x80) {
            out.push_back(c);
            i += 1;
        } else if ((c >> 5) == 0x6 && i + 1 < n) {
            char32_t cp = ((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F);
            out.push_back(cp);
            i += 2;
        } else if ((c >> 4) == 0xE && i + 2 < n) {
            char32_t cp = ((c & 0x0F) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                          (static_cast<uint8_t>(s[i + 2]) & 0x3F);
            out.push_back(cp);
            i += 3;
        } else if ((c >> 3) == 0x1E && i + 3 < n) {
            char32_t cp = ((c & 0x07) << 18) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                          ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) | (static_cast<uint8_t>(s[i + 3]) & 0x3F);
            out.push_back(cp);
            i += 4;
        } else {
            i += 1;
        }
    }
    return out;
}

inline void append_u32_to_utf8(char32_t cp, std::string & out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

inline std::string u32_to_utf8(const std::u32string & s) {
    std::string out;
    out.reserve(s.size() * 3);
    for (char32_t cp : s) append_u32_to_utf8(cp, out);
    return out;
}

inline std::string cp_to_utf8(char32_t cp) {
    std::string out;
    append_u32_to_utf8(cp, out);
    return out;
}

// Canonical NFC normalization for Devanagari singleton decompositions + IAST Latin precomposed characters
inline std::u32string nfc_normalize_u32(const std::u32string & in) {
    std::u32string out;
    out.reserve(in.size() + 4);
    for (size_t i = 0; i < in.size(); ++i) {
        char32_t c = in[i];
        // Devanagari singleton decompositions in Unicode NFC
        switch (c) {
            case 0x0929: out.push_back(0x0928); out.push_back(0x093C); continue;
            case 0x0931: out.push_back(0x0930); out.push_back(0x093C); continue;
            case 0x0934: out.push_back(0x0933); out.push_back(0x093C); continue;
            case 0x0958: out.push_back(0x0915); out.push_back(0x093C); continue;
            case 0x0959: out.push_back(0x0916); out.push_back(0x093C); continue;
            case 0x095A: out.push_back(0x0917); out.push_back(0x093C); continue;
            case 0x095B: out.push_back(0x091C); out.push_back(0x093C); continue;
            case 0x095C: out.push_back(0x0921); out.push_back(0x093C); continue;
            case 0x095D: out.push_back(0x0922); out.push_back(0x093C); continue;
            case 0x095E: out.push_back(0x092B); out.push_back(0x093C); continue;
            case 0x095F: out.push_back(0x092F); out.push_back(0x093C); continue;
            default: break;
        }
        out.push_back(c);
    }
    return out;
}

inline std::string nfc_normalize(const std::string & s) {
    return u32_to_utf8(nfc_normalize_u32(utf8_to_u32(s)));
}

inline bool u32_starts_with(const std::u32string & s, const std::u32string & pfx) {
    return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

inline bool u32_ends_with(const std::u32string & s, const std::u32string & sfx) {
    return s.size() >= sfx.size() && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
}

inline bool str_ends_with(const std::string & s, const std::string & sfx) {
    return s.size() >= sfx.size() && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
}

inline bool str_starts_with(const std::string & s, const std::string & pfx) {
    return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

inline std::string trim_str(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    return s.substr(a, b - a);
}

inline void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

inline double py_round4(double val) {
    // Python round(x, 4) round-half-to-even
    return std::nearbyint(val * 10000.0) / 10000.0;
}

inline double chandas_round4(double x) {
    return x >= 0.0 ? std::floor(x * 10000.0 + 0.5) / 10000.0 : std::ceil(x * 10000.0 - 0.5) / 10000.0;
}

inline int js_round(double x) {
    return static_cast<int>(std::floor(x + 0.5));
}

// ---------------- Symbols (sakhiya/symbols.py) ----------------
inline const std::vector<std::string> & get_symbols() {
    static const std::vector<std::string> syms = {
        "_", "|", "<s>", "</s>",
        ",", ".", "?", "!", ";",
        "k", "kʰ", "g", "gʱ", "ŋ", "ts", "tsʰ", "dz", "dzʱ", "ɲ",
        "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ", "t̪", "t̪ʰ", "d̪", "d̪ʱ", "n",
        "p", "pʰ", "b", "bʱ", "m", "j", "r", "l", "w", "ʃ", "s", "h", "ɽ", "ɽʱ", "z", "f",
        "ʌ", "a", "i", "u", "e", "o", "ɛ", "ɔ",
        "ʌ̃", "ã", "ĩ", "ũ", "ẽ", "õ", "ɛ̃", "ɔ̃"
    };
    return syms;
}

inline const std::unordered_map<std::string, int> & get_sym2id() {
    static std::unordered_map<std::string, int> m;
    if (m.empty()) {
        const auto & s = get_symbols();
        for (size_t i = 0; i < s.size(); ++i) m[s[i]] = static_cast<int>(i);
    }
    return m;
}

inline bool is_pause_id(int64_t id) {
    // "|" (1), "," (4), "." (5), "?" (6), "!" (7), ";" (8)
    return id == 1 || (id >= 4 && id <= 8);
}

inline bool is_punct_phone(const std::string & p) {
    return p == "|" || p == "," || p == "." || p == "?" || p == "!" || p == ";";
}

inline bool starts_with_vowel_base(const std::string & p) {
    if (p.empty()) return false;
    if (str_starts_with(p, "ʌ") || str_starts_with(p, "a") || str_starts_with(p, "i") ||
        str_starts_with(p, "u") || str_starts_with(p, "e") || str_starts_with(p, "o")) {
        return true;
    }
    return false;
}

// ---------------- Numbers (sakhiya/numbers.py) ----------------
inline const std::vector<std::string> & get_num_u() {
    static const std::vector<std::string> u = {
        "शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ",
        "दश", "एघार", "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र", "सत्र", "अठार", "उन्नाइस",
        "बिस", "एक्काइस", "बाइस", "तेइस", "चौबिस", "पच्चिस", "छब्बिस", "सत्ताइस", "अठ्ठाइस", "उनन्तिस",
        "तिस", "एकतिस", "बत्तिस", "तेत्तिस", "चौँतिस", "पैँतिस", "छत्तिस", "सैँतिस", "अठतिस", "उनन्चालिस",
        "चालिस", "एकचालिस", "बयालिस", "त्रिचालिस", "चवालिस", "पैँतालिस", "छयालिस", "सतचालिस", "अठचालिस", "उनन्चास",
        "पचास", "एकाउन्न", "बाउन्न", "त्रिपन्न", "चौवन्न", "पचपन्न", "छपन्न", "सन्ताउन्न", "अन्ठाउन्न", "उनान्न",
        "साठी", "एकसट्ठी", "बैसट्ठी", "त्रिसट्ठी", "चौँसट्ठी", "पैँसट्ठी", "छैसट्ठी", "सतसट्ठी", "अठसट्ठी", "उनान्सत्तरी",
        "सत्तरी", "एकहत्तर", "बहत्तर", "त्रिहत्तर", "चौहत्तर", "पचहत्तर", "छयाहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
        "असी", "एकासी", "बयासी", "त्रियासी", "चौरासी", "पचासी", "छयासी", "सतासी", "अठासी", "उनान्नब्बे",
        "नब्बे", "एकान्नब्बे", "बयान्नब्बे", "त्रियान्नब्बे", "चौरान्नब्बे", "पन्चान्नब्बे", "छयान्नब्बे", "सन्तान्नब्बे", "अन्ठान्नब्बे", "उनान्सय"
    };
    return u;
}

inline std::string num_words(int64_t n) {
    const auto & u = get_num_u();
    if (n < 0) n = -n;
    if (n < 100) return u[static_cast<size_t>(n)];
    static const std::pair<int64_t, const char *> big[] = {
        {100000000000LL, "खर्ब"},
        {1000000000LL, "अर्ब"},
        {10000000LL, "करोड"},
        {100000LL, "लाख"},
        {1000LL, "हजार"},
        {100LL, "सय"}
    };
    for (const auto & kv : big) {
        if (n >= kv.first) {
            int64_t q = n / kv.first;
            int64_t r = n % kv.first;
            std::string head = num_words(q) + " " + kv.second;
            return r ? (head + " " + num_words(r)) : head;
        }
    }
    return "";
}

inline std::string digits_words(const std::string & s) {
    const auto & u = get_num_u();
    std::string out;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            if (!out.empty()) out += " ";
            out += u[static_cast<size_t>(c - '0')];
        }
    }
    return out;
}

inline std::string expand_number_token(std::string tok) {
    tok.erase(std::remove(tok.begin(), tok.end(), ','), tok.end());
    auto dot = tok.find('.');
    if (dot != std::string::npos) {
        std::string a = tok.substr(0, dot);
        std::string b = tok.substr(dot + 1);
        int64_t va = a.empty() ? 0 : std::stoll(a);
        return num_words(va) + " दशमलव " + digits_words(b);
    }
    if (tok.size() > 1 && tok[0] == '0') return digits_words(tok);
    if (tok.size() > 13) return digits_words(tok);
    if (tok.empty()) return "";
    return num_words(std::stoll(tok));
}

inline bool is_ascii_digit(char32_t c) { return c >= U'0' && c <= U'9'; }
inline bool is_ascii_upper(char32_t c) { return c >= U'A' && c <= U'Z'; }
inline bool is_ascii_alpha(char32_t c) { return (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z'); }
inline bool is_word_cp(char32_t c) {
    return is_ascii_alpha(c) || is_ascii_digit(c) || c == U'_' || (c >= 0x0900 && c <= 0x097F);
}

inline std::string expand_numbers(const std::string & text) {
    static const char * months[12] = {
        "बैशाख", "जेठ", "असार", "श्रावण", "भदौ", "असोज",
        "कात्तिक", "मंसिर", "पौष", "माघ", "फागुन", "चैत"
    };
    static const std::unordered_map<char, std::string> letters = {
        {'a',"ए"},{'b',"बी"},{'c',"सी"},{'d',"डी"},{'e',"इ"},{'f',"एफ"},{'g',"जी"},{'h',"एच"},
        {'i',"आई"},{'j',"जे"},{'k',"के"},{'l',"एल"},{'m',"एम"},{'n',"एन"},{'o',"ओ"},{'p',"पी"},
        {'q',"क्यु"},{'r',"आर"},{'s',"एस"},{'t',"टी"},{'u',"यु"},{'v',"भी"},{'w',"डब्लु"},{'x',"एक्स"},
        {'y',"वाई"},{'z',"जेड"}
    };

    // 1. Translate Devanagari digits ०-९ to 0-9
    std::u32string u = utf8_to_u32(text);
    for (char32_t & c : u) {
        if (c >= 0x0966 && c <= 0x096F) c = U'0' + (c - 0x0966);
    }

    // Helper to extract number string starting at index i: \d[\d,]*(?:\.\d+)?
    auto match_num = [&](const std::u32string & s, size_t i, size_t & end_i) -> std::string {
        if (i >= s.size() || !is_ascii_digit(s[i])) return "";
        size_t j = i + 1;
        while (j < s.size() && (is_ascii_digit(s[j]) || s[j] == U',')) ++j;
        if (j + 1 < s.size() && s[j] == U'.' && is_ascii_digit(s[j + 1])) {
            j += 2;
            while (j < s.size() && is_ascii_digit(s[j])) ++j;
        }
        end_i = j;
        std::string res;
        for (size_t k = i; k < j; ++k) res.push_back(static_cast<char>(s[k]));
        return res;
    };

    // Pass A: dates \b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b and times \b(\d{1,2}):(\d{2})\b
    std::u32string u2;
    for (size_t i = 0; i < u.size();) {
        bool prev_b = (i == 0 || !is_word_cp(u[i - 1]));
        if (prev_b && i + 7 < u.size() && is_ascii_digit(u[i]) && is_ascii_digit(u[i + 1]) &&
            is_ascii_digit(u[i + 2]) && is_ascii_digit(u[i + 3]) && (u[i + 4] == U'-' || u[i + 4] == U'/')) {
            size_t p = i + 5;
            std::string mo_s, d_s;
            while (p < u.size() && is_ascii_digit(u[p]) && mo_s.size() < 2) mo_s.push_back(static_cast<char>(u[p++]));
            if (!mo_s.empty() && p < u.size() && (u[p] == U'-' || u[p] == U'/')) {
                ++p;
                while (p < u.size() && is_ascii_digit(u[p]) && d_s.size() < 2) d_s.push_back(static_cast<char>(u[p++]));
                bool next_b = (p == u.size() || !is_word_cp(u[p]));
                if (!d_s.empty() && next_b) {
                    int y = (u[i] - U'0') * 1000 + (u[i + 1] - U'0') * 100 + (u[i + 2] - U'0') * 10 + (u[i + 3] - U'0');
                    int mo = std::stoi(mo_s);
                    int d = std::stoi(d_s);
                    if (mo >= 1 && mo <= 12 && d >= 1 && d <= 32) {
                        std::string rep = num_words(y) + " साल " + months[mo - 1] + " " + num_words(d) + " गते";
                        auto r32 = utf8_to_u32(rep);
                        u2.insert(u2.end(), r32.begin(), r32.end());
                        i = p;
                        continue;
                    }
                }
            }
        }
        // Time: \b(\d{1,2}):(\d{2})\b
        if (prev_b && is_ascii_digit(u[i])) {
            size_t p = i;
            std::string h_s;
            while (p < u.size() && is_ascii_digit(u[p]) && h_s.size() < 2) h_s.push_back(static_cast<char>(u[p++]));
            if (!h_s.empty() && p + 2 < u.size() && u[p] == U':' && is_ascii_digit(u[p + 1]) && is_ascii_digit(u[p + 2])) {
                size_t end_p = p + 3;
                bool next_b = (end_p == u.size() || !is_word_cp(u[end_p]));
                if (next_b) {
                    int h = std::stoi(h_s);
                    int mi = (u[p + 1] - U'0') * 10 + (u[p + 2] - U'0');
                    std::string rep = mi ? (num_words(h) + " बजेर " + num_words(mi) + " मिनेट") : (num_words(h) + " बजे");
                    auto r32 = utf8_to_u32(rep);
                    u2.insert(u2.end(), r32.begin(), r32.end());
                    i = end_p;
                    continue;
                }
            }
        }
        u2.push_back(u[i++]);
    }
    u.swap(u2);
    u2.clear();

    // Pass B: Currency (?:रु\.?|रू\.?|Rs\.?|NPR)\s*(\d[\d,]*(?:\.\d+)?)
    static const std::vector<std::u32string> curr_pfxs = {
        U"रु.", U"रु", U"रू.", U"रू", U"Rs.", U"Rs", U"NPR"
    };
    for (size_t i = 0; i < u.size();) {
        bool matched_curr = false;
        for (const auto & pfx : curr_pfxs) {
            if (i + pfx.size() <= u.size() && u.compare(i, pfx.size(), pfx) == 0) {
                size_t p = i + pfx.size();
                while (p < u.size() && (u[p] == U' ' || u[p] == U'\t')) ++p;
                size_t end_num = p;
                std::string num_s = match_num(u, p, end_num);
                if (!num_s.empty()) {
                    std::string rep = expand_number_token(num_s) + " रुपैयाँ";
                    auto r32 = utf8_to_u32(rep);
                    u2.insert(u2.end(), r32.begin(), r32.end());
                    i = end_num;
                    matched_curr = true;
                    break;
                }
            }
        }
        if (!matched_curr) u2.push_back(u[i++]);
    }
    u.swap(u2);
    u2.clear();

    // Pass C: Percent (\d[\d,]*(?:\.\d+)?)\s*%, km (\d[\d,]*(?:\.\d+)?)\s*(?:किमी|km)\b, and bare numbers
    for (size_t i = 0; i < u.size();) {
        if (is_ascii_digit(u[i])) {
            size_t end_num = i;
            std::string num_s = match_num(u, i, end_num);
            if (!num_s.empty()) {
                size_t p = end_num;
                while (p < u.size() && (u[p] == U' ' || u[p] == U'\t')) ++p;
                if (p < u.size() && u[p] == U'%') {
                    std::string rep = expand_number_token(num_s) + " प्रतिशत";
                    auto r32 = utf8_to_u32(rep);
                    u2.insert(u2.end(), r32.begin(), r32.end());
                    i = p + 1;
                    continue;
                }
                if (p + 4 <= u.size() && u.compare(p, 4, U"किमी") == 0 && (p + 4 == u.size() || !is_word_cp(u[p + 4]))) {
                    std::string rep = expand_number_token(num_s) + " किलोमिटर";
                    auto r32 = utf8_to_u32(rep);
                    u2.insert(u2.end(), r32.begin(), r32.end());
                    i = p + 4;
                    continue;
                }
                if (p + 2 <= u.size() && u.compare(p, 2, U"km") == 0 && (p + 2 == u.size() || !is_word_cp(u[p + 2]))) {
                    std::string rep = expand_number_token(num_s) + " किलोमिटर";
                    auto r32 = utf8_to_u32(rep);
                    u2.insert(u2.end(), r32.begin(), r32.end());
                    i = p + 2;
                    continue;
                }
                while (!num_s.empty() && num_s.back() == ',') num_s.pop_back();
                std::string rep = expand_number_token(num_s);
                auto r32 = utf8_to_u32(rep);
                u2.insert(u2.end(), r32.begin(), r32.end());
                i = end_num;
                continue;
            }
        }
        u2.push_back(u[i++]);
    }
    u.swap(u2);
    u2.clear();

    // Pass D: Acronyms \b([A-Z]{2,5})\b
    for (size_t i = 0; i < u.size();) {
        bool prev_b = (i == 0 || !is_word_cp(u[i - 1]));
        if (prev_b && is_ascii_upper(u[i])) {
            size_t j = i;
            while (j < u.size() && is_ascii_upper(u[j])) ++j;
            size_t len = j - i;
            bool next_b = (j == u.size() || !is_word_cp(u[j]));
            if (len >= 2 && len <= 5 && next_b) {
                std::string rep;
                for (size_t k = i; k < j; ++k) {
                    if (!rep.empty()) rep += " ";
                    char lc = static_cast<char>(u[k] - U'A' + 'a');
                    auto it = letters.find(lc);
                    if (it != letters.end()) rep += it->second;
                }
                auto r32 = utf8_to_u32(rep);
                u2.insert(u2.end(), r32.begin(), r32.end());
                i = j;
                continue;
            }
        }
        u2.push_back(u[i++]);
    }

    return u32_to_utf8(u2);
}

// ---------------- Nepali G2P & Translit (sakhiya/g2p.py + sakhiya/translit.py) ----------------
inline const std::unordered_map<char32_t, std::string> & get_deva_c() {
    static const std::unordered_map<char32_t, std::string> m = {
        {U'क',"k"},{U'ख',"kʰ"},{U'ग',"g"},{U'घ',"gʱ"},{U'ङ',"ŋ"},{U'च',"ts"},{U'छ',"tsʰ"},{U'ज',"dz"},{U'झ',"dzʱ"},
        {U'ञ',"ɲ"},{U'ट',"ʈ"},{U'ळ',"l"},{U'ठ',"ʈʰ"},{U'ड',"ɖ"},{U'ढ',"ɖʱ"},{U'ण',"ɳ"},{U'त',"t̪"},{U'थ',"t̪ʰ"},{U'द',"d̪"},
        {U'ध',"d̪ʱ"},{U'न',"n"},{U'प',"p"},{U'फ',"pʰ"},{U'ब',"b"},{U'भ',"bʱ"},{U'म',"m"},{U'य',"j"},{U'र',"r"},
        {U'ल',"l"},{U'व',"w"},{U'श',"ʃ"},{U'ष',"ʃ"},{U'स',"s"},{U'ह',"h"}
    };
    return m;
}

inline const std::unordered_map<char32_t, std::string> & get_nukta_map() {
    static const std::unordered_map<char32_t, std::string> m = {
        {U'ड',"ɽ"},{U'ढ',"ɽʱ"},{U'ज',"dz"},{U'फ',"pʰ"},{U'क',"k"},{U'ख',"kʰ"},{U'ग',"g"}
    };
    return m;
}

inline const std::unordered_map<char32_t, std::string> & get_iv() {
    static const std::unordered_map<char32_t, std::string> m = {
        {U'अ',"ʌ"},{U'आ',"a"},{U'इ',"i"},{U'ई',"i"},{U'उ',"u"},{U'ऊ',"u"},{U'ए',"e"},{U'ओ',"o"},{U'ऑ',"o"}
    };
    return m;
}

inline const std::unordered_map<char32_t, std::vector<std::string>> & get_iv2() {
    static const std::unordered_map<char32_t, std::vector<std::string>> m = {
        {U'ऐ',{"ʌ","i"}},{U'औ',{"ʌ","u"}},{U'ऋ',{"r","i"}}
    };
    return m;
}

inline const std::unordered_map<char32_t, std::string> & get_matra() {
    static const std::unordered_map<char32_t, std::string> m = {
        {U'ा',"a"},{U'ि',"i"},{U'ी',"i"},{U'ु',"u"},{U'ू',"u"},{U'े',"e"},{U'ो',"o"},{U'ॉ',"o"}
    };
    return m;
}

inline const std::unordered_map<char32_t, std::vector<std::string>> & get_matra2() {
    static const std::unordered_map<char32_t, std::vector<std::string>> m = {
        {U'ै',{"ʌ","i"}},{U'ौ',{"ʌ","u"}},{U'ृ',{"r","i"}}
    };
    return m;
}

inline const std::unordered_map<char32_t, std::string> & get_nasal_place() {
    static const std::unordered_map<char32_t, std::string> m = {
        {U'क',"ŋ"},{U'ख',"ŋ"},{U'ग',"ŋ"},{U'घ',"ŋ"},
        {U'च',"n"},{U'छ',"n"},{U'ज',"n"},{U'झ',"n"},
        {U'ट',"n"},{U'ठ',"n"},{U'ड',"n"},{U'ढ',"n"},
        {U'त',"n"},{U'थ',"n"},{U'द',"n"},{U'ध',"n"},
        {U'प',"m"},{U'फ',"m"},{U'ब',"m"},{U'भ',"m"}
    };
    return m;
}

struct G2PUnit {
    std::string ph;
    char kind; // 'C', 'V', 'S'
};

inline std::vector<G2PUnit> parse_word_units(const std::u32string & w) {
    const auto & C = get_deva_c();
    const auto & NUKTA = get_nukta_map();
    const auto & IV = get_iv();
    const auto & IV2 = get_iv2();
    const auto & MATRA = get_matra();
    const auto & MATRA2 = get_matra2();
    const auto & NASAL = get_nasal_place();

    std::vector<G2PUnit> out;
    auto nasalize = [&]() {
        for (int k = static_cast<int>(out.size()) - 1; k >= 0; --k) {
            if (out[k].kind == 'V' || out[k].kind == 'S') {
                std::string ph = (out[k].ph == "ʌ?") ? "ʌ" : out[k].ph;
                if (!str_ends_with(ph, "\xCC\x83")) ph += "\xCC\x83";
                out[k] = {ph, 'V'};
                if ((ph == "i" || ph == "u" || ph == "i\xCC\x83" || ph == "u\xCC\x83") &&
                    k >= 1 && (out[k - 1].kind == 'V' || out[k - 1].kind == 'S')) {
                    std::string p0 = (out[k - 1].ph == "ʌ?") ? "ʌ" : out[k - 1].ph;
                    if (p0 == "ʌ" || p0 == "a" || p0 == "o" || p0 == "e") {
                        out[k - 1] = {p0 + "\xCC\x83", 'V'};
                    }
                }
                return;
            }
        }
    };

    size_t i = 0, n = w.size();
    while (i < n) {
        char32_t ch = w[i];
        auto itC = C.find(ch);
        if (itC != C.end()) {
            std::string ph = itC->second;
            if (i + 1 < n && w[i + 1] == 0x093C) { // NUK
                auto itN = NUKTA.find(ch);
                if (itN != NUKTA.end()) ph = itN->second;
                ++i;
            }
            out.push_back({ph, 'C'});
            ++i;
            if (i < n && w[i] == 0x094D) { // HAL
                ++i;
            } else if (i < n && (MATRA.count(w[i]) || MATRA2.count(w[i]))) {
                auto itM = MATRA.find(w[i]);
                if (itM != MATRA.end()) {
                    out.push_back({itM->second, 'V'});
                } else {
                    for (const auto & p : MATRA2.at(w[i])) out.push_back({p, 'V'});
                }
                ++i;
            } else {
                out.push_back({"ʌ?", 'S'});
            }
        } else if (IV.count(ch)) {
            std::string v_ph = IV.at(ch);
            if (i > 0 && !out.empty() && out.back().kind == 'V') {
                std::string prev = out.back().ph;
                if (str_ends_with(prev, "\xCC\x83")) prev.resize(prev.size() - 2);
                if (prev == v_ph) out.push_back({"|", 'C'});
            }
            out.push_back({v_ph, 'V'});
            ++i;
        } else if (IV2.count(ch)) {
            std::vector<std::string> seq = IV2.at(ch);
            if ((ch == U'औ' || ch == U'ऐ') && !out.empty() && out.back().kind == 'V' &&
                (out.back().ph == "a" || out.back().ph == "a\xCC\x83")) {
                seq.erase(seq.begin());
            }
            for (const auto & p : seq) out.push_back({p, p == "r" ? 'C' : 'V'});
            ++i;
        } else if (ch == 0x0901) { // CHAN ँ
            nasalize();
            ++i;
        } else if (ch == 0x0902) { // ANU ं
            char32_t nxt = (i + 1 < n) ? w[i + 1] : 0;
            auto itNP = NASAL.find(nxt);
            if (itNP != NASAL.end()) {
                if (!out.empty() && out.back().kind == 'S') out.back() = {"ʌ", 'V'};
                out.push_back({itNP->second, 'C'});
            } else {
                nasalize();
            }
            ++i;
        } else if (ch == 0x0950) { // ॐ
            out.push_back({"o", 'V'});
            out.push_back({"m", 'C'});
            ++i;
        } else if (ch == 0x0903) { // VIS ः
            out.push_back({"h", 'C'});
            ++i;
        } else {
            ++i;
        }
    }
    return out;
}

inline bool keep_final_schwa(const std::u32string & w32, const std::string & w_utf8,
                             const std::vector<G2PUnit> & units, size_t idx, size_t n_syl) {
    static const std::unordered_set<std::string> loan_no_schwa = {
        "टेक्स्ट", "टेस्ट", "पोस्ट", "स्पीच", "फास्ट", "लिस्ट", "कन्टेन्ट", "प्रिन्ट", "ब्रान्ड", "वर्ल्ड",
        "फिल्म", "इन्टेलिजेन्स", "एम्बुलेन्स", "जेम्स", "फ्रान्स", "पार्क", "बसपार्क", "वार्ड", "कार्ड", "बोर्ड", "एण्ड", "सरपंच"
    };
    static const std::vector<std::string> loan_final_cl = {
        "क्स्ट", "स्ट", "स्क", "स्प", "क्स", "ल्ड", "न्ड", "न्ट", "क्ट", "न्स", "ल्म", "म्स", "र्क", "र्ड"
    };
    static const std::unordered_set<std::string> keep_final = {
        "आज", "बाट", "सँग", "संग", "तल", "तर", "तिर", "जब", "तब", "अब", "अझ", "वर", "पर", "सुख", "दुख", "दुःख",
        "बाहिर", "भित्र", "हेर", "गर", "भन", "सुन", "पढ", "लेख", "राख", "छैन", "होइन", "सय", "समय", "निर्णय",
        "प्राय", "समुदाय", "उपाय", "बिहान", "सकल", "रुन", "पख", "साँझपख", "विषय", "परिचय", "आशय", "हृदय", "महोदय", "न्याय", "अध्याय",
        "खान", "जान", "लिन", "हुन", "रहन", "आउन", "पाउन", "गाउन", "लाउन", "धाउन", "छाउन",
        "गरेर", "भनेर", "हेरेर", "आएर", "गएर", "खाएर", "लिएर", "दिएर", "बसेर", "सुनेर", "पढेर"
    };
    static const std::vector<std::string> keep_sfx1 = {
        "बाट", "सँग", "संग", "तिर", "तल", "बाहिर", "भित्र", "ीय", "प्रिय", "ष्ट्रिय", "लय", "दय", "शय", "पख"
    };
    static const std::vector<std::string> keep_sfx2 = {
        "उन", "एर", "ेर", "दैन", "एन", "छैन", "होइन"
    };

    if (!w_utf8.empty()) {
        if (loan_no_schwa.count(w_utf8)) return false;
        for (const auto & cl : loan_final_cl) {
            if (str_ends_with(w_utf8, cl)) return false;
        }
    }
    if (n_syl == 1) return true;
    if (idx >= 2 && units[idx - 1].kind == 'C' && units[idx - 2].kind == 'C') return true;
    if (w_utf8.empty()) return false;
    if (!w32.empty() && w32.back() == U'छ') return true;
    if (keep_final.count(w_utf8)) return true;
    for (const auto & sfx : keep_sfx1) {
        if (str_ends_with(w_utf8, sfx)) return true;
    }
    if (w32.size() >= 3) {
        for (const auto & sfx : keep_sfx2) {
            if (str_ends_with(w_utf8, sfx)) return true;
        }
    }
    return false;
}

inline std::vector<std::string> resolve_units(const std::vector<G2PUnit> & units,
                                              const std::u32string & w32, const std::string & w_utf8) {
    size_t n_syl = 0;
    for (const auto & u : units) {
        if (u.kind == 'V' || u.kind == 'S') ++n_syl;
    }
    std::vector<std::string> res;
    for (size_t idx = 0; idx < units.size(); ++idx) {
        if (units[idx].kind != 'S') {
            res.push_back(units[idx].ph);
            continue;
        }
        bool last = (idx == units.size() - 1);
        for (size_t j = idx + 1; j < units.size(); ++j) {
            if (units[j].kind != 'C') last = false;
        }
        if (!last) {
            res.push_back("ʌ");
        } else if (keep_final_schwa(w32, w_utf8, units, idx, n_syl)) {
            res.push_back("ʌ");
        }
    }
    return res;
}

inline std::vector<std::string> honorific_glide(const std::vector<std::string> & ph) {
    std::vector<std::string> out;
    for (size_t i = 0; i < ph.size(); ++i) {
        if (ph[i] == "e" && i >= 3 && ph[i - 3] == "u" && ph[i - 2] == "bʱ" && ph[i - 1] == "ʌ") {
            out.push_back("j");
        }
        out.push_back(ph[i]);
    }
    return out;
}

inline std::vector<std::string> word_core(const std::u32string & w32, const std::string & w_utf8) {
    auto ph = resolve_units(parse_word_units(w32), w32, w_utf8);
    if (w_utf8.find("नुभ") != std::string::npos) return honorific_glide(ph);
    return ph;
}

inline const std::unordered_map<std::string, std::vector<std::string>> & get_g2p_lex() {
    static const std::unordered_map<std::string, std::vector<std::string>> lex = {
        {"काठमाडौँ", {"k", "a", "ʈʰ", "m", "a", "ɖ", "ʌ̃", "ũ"}},
        {"काठमाडौं", {"k", "a", "ʈʰ", "m", "a", "ɖ", "ʌ̃", "ũ"}},
        {"माछापुच्छ्रे", {"m", "a", "tsʰ", "a", "p", "u", "tsʰ", "r", "e"}},
        {"साँझपख", {"s", "ã", "dzʱ", "p", "ʌ", "kʰ", "ʌ"}},
        {"वाह", {"w", "a", "h"}},
        {"आआफ्नो", {"a", "|", "a", "pʰ", "n", "o"}},
        {"आआफ्ना", {"a", "|", "a", "pʰ", "n", "a"}},
        {"आआफ्नै", {"a", "|", "a", "pʰ", "n", "ʌ", "i"}},
        {"राजधानी", {"r", "a", "dz", "d̪ʱ", "a", "n", "i"}},
        {"प्रधानमन्त्री", {"p", "r", "ʌ", "d̪ʱ", "a", "n", "m", "ʌ", "n", "t̪", "r", "i"}},
        {"सगरमाथा", {"s", "ʌ", "g", "ʌ", "r", "m", "a", "t̪ʰ", "a"}},
        {"सुदूरपश्चिम", {"s", "u", "d̪", "u", "r", "p", "ʌ", "ʃ", "ts", "i", "m"}},
        {"लगभग", {"l", "ʌ", "g", "bʱ", "ʌ", "g"}},
        {"फुटबल", {"pʰ", "u", "ʈ", "b", "ʌ", "l"}},
        {"इन्टरनेट", {"i", "n", "ʈ", "ʌ", "r", "n", "e", "ʈ"}},
        {"बसपार्क", {"b", "ʌ", "s", "p", "a", "r", "k"}},
        {"वेबसाइट", {"w", "e", "b", "s", "a", "i", "ʈ"}},
        {"रेलवे", {"r", "e", "l", "w", "e"}},
        {"देवराज", {"d̪", "e", "w", "r", "a", "dz"}},
        {"डडेलधुरा", {"ɖ", "ʌ", "ɖ", "e", "l", "d̪ʱ", "u", "r", "a"}},
        {"सिलसिला", {"s", "i", "l", "s", "i", "l", "a"}},
        {"कोबोर्ड", {"k", "o", "b", "o", "r", "ɖ"}},
        {"किचकवध", {"k", "i", "ts", "ʌ", "k", "w", "ʌ", "d̪ʱ"}},
        {"सुरतगढ", {"s", "u", "r", "ʌ", "t̪", "g", "ʌ", "ɖʱ"}},
        {"किमडाँडा", {"k", "i", "m", "ɖ", "ã", "ɖ", "a"}},
        {"अर्घाखाँची", {"ʌ", "r", "gʱ", "a", "kʰ", "ã", "ts", "i"}},
        {"फिल्मसम्बन्धि", {"pʰ", "i", "l", "m", "s", "ʌ", "m", "b", "ʌ", "n", "d̪ʱ", "i"}},
        {"फ्रान्स", {"pʰ", "r", "a", "n", "s"}},
        {"एण्ड", {"e", "n", "ɖ"}},
        {"एकदम", {"e", "k", "d̪", "ʌ", "m"}},
        {"कलकल", {"k", "ʌ", "l", "k", "ʌ", "l"}},
        {"चिरबिर", {"ts", "i", "r", "b", "i", "r"}},
        {"किनमेल", {"k", "i", "n", "m", "e", "l"}},
        {"फलफूल", {"pʰ", "ʌ", "l", "pʰ", "u", "l"}},
        {"कागजात", {"k", "a", "g", "dz", "a", "t̪"}},
        {"जनसंख्या", {"dz", "ʌ", "n", "s", "ʌ", "ŋ", "kʰ", "j", "a"}},
        {"तापक्रम", {"t̪", "a", "p", "k", "r", "ʌ", "m"}},
        {"अधिकतम", {"ʌ", "d̪ʱ", "i", "k", "t̪", "ʌ", "m"}},
        {"अठहत्तर", {"ʌ", "ʈʰ", "h", "ʌ", "t̪", "t̪", "ʌ", "r"}},
        {"नेक्स्टजेन", {"n", "e", "k", "s", "ʈ", "dz", "e", "n"}}
    };
    return lex;
}

inline const std::vector<std::u32string> & get_suffixes_u32() {
    static std::vector<std::u32string> sfx;
    if (sfx.empty()) {
        const char * raw =
            "हरूलाई हरूले हरूको हरूका हरूमा हरूबाट हरूसँग हरूतिर हरूभित्र हरूबाहिर हरूमाथि हरू हरुलाई हरुले हरुको हरुका हरुमा हरुबाट हरुसँग हरुतिर हरु "
            "ज्यूहरूलाई ज्यूहरूले ज्यूहरूको ज्यूलाई ज्यूले ज्यूको ज्यूका ज्यूमा ज्यूबाट ज्यूसँग ज्यू ज्युलाई ज्युले ज्युको ज्यु हुक्रे हुक्रेन "
            "लाई बाटै बाट देखि सम्म सँगै सँग संग सँगे संगे ले को का की मा मे से के हेन सन लग केर कर पछि अघि पहिले "
            "भित्र बाहिर माथि तल नजिक भरि भर पट्टि पटि जस्तो जस्तै जति मात्र पनि नै चाहिँ चाहिं पिच्छे वरिपरि बारे विरुद्ध अनुसार समेत बित्तिकै अगाडि पछाडि प्रति तर्फ पूर्वक घर स्थान स्थल पत्र";
        std::istringstream iss(raw);
        std::string tok;
        while (iss >> tok) sfx.push_back(utf8_to_u32(tok));
        std::stable_sort(sfx.begin(), sfx.end(), [](const std::u32string & a, const std::u32string & b) {
            return a.size() > b.size();
        });
    }
    return sfx;
}

inline std::vector<std::string> word_to_phones(const std::string & w_in) {
    std::u32string w32 = nfc_normalize_u32(utf8_to_u32(w_in));
    std::string w = u32_to_utf8(w32);
    const auto & lex = get_g2p_lex();
    auto it = lex.find(w);
    if (it != lex.end()) return it->second;

    const auto & C = get_deva_c();
    for (const auto & sfx : get_suffixes_u32()) {
        if (w32.size() > sfx.size() && u32_ends_with(w32, sfx) && C.count(sfx[0])) {
            std::u32string stem32 = w32.substr(0, w32.size() - sfx.size());
            std::string stem = u32_to_utf8(stem32);
            if (lex.count(stem) || (!stem32.empty() && C.count(stem32.back())) ||
                u32_ends_with(stem32, U"हरू") || u32_ends_with(stem32, U"हरु") ||
                u32_ends_with(stem32, U"ज्यू") || u32_ends_with(stem32, U"ज्यु")) {
                auto a = word_to_phones(stem);
                auto b = word_to_phones(u32_to_utf8(sfx));
                a.insert(a.end(), b.begin(), b.end());
                return a;
            }
            break;
        }
    }
    return word_core(w32, w);
}

// Romanized Nepali -> Devanagari (sakhiya/translit.py)
struct TranslitTables {
    std::unordered_map<std::string, std::string> ne;
    std::unordered_map<std::string, std::string> direct;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> thr_exact;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> thr_key;
    bool loaded = false;
};

inline std::vector<std::string> roman_tok(const std::string & r_in) {
    static const std::unordered_set<std::string> rc_set = {
        "chh","ch","kh","gh","jh","th","dh","ph","bh","sh","k","g","c","j","t","d","n","p","b","m","y","r","l","w","v","s","h","f","z","q","x"
    };
    static const std::unordered_set<std::string> rv_set = {
        "aa","ai","au","ee","ii","oo","uu","a","i","u","e","o"
    };
    std::string r;
    r.reserve(r_in.size());
    for (char c : r_in) r.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    std::vector<std::string> out;
    size_t i = 0;
    while (i < r.size()) {
        bool hit = false;
        for (size_t L : {3, 2, 1}) {
            if (i + L <= r.size()) {
                std::string seg = r.substr(i, L);
                if ((L == 3 && seg == "chh") || (L <= 2 && (rc_set.count(seg) || rv_set.count(seg)))) {
                    out.push_back(seg);
                    i += L;
                    hit = true;
                    break;
                }
            }
        }
        if (!hit) ++i;
    }
    return out;
}

inline std::string roman_key(const std::string & r_in) {
    static const std::unordered_map<std::string, std::string> rc_ph = {
        {"chh","tsʰ"},{"ch","ts"},{"kh","kʰ"},{"gh","gʱ"},{"jh","dzʱ"},{"th","t̪ʰ"},{"dh","d̪ʱ"},{"ph","pʰ"},{"bh","bʱ"},
        {"sh","ʃ"},{"k","k"},{"g","g"},{"c","k"},{"j","dz"},{"t","t̪"},{"d","d̪"},{"n","n"},{"p","p"},{"b","b"},
        {"m","m"},{"y","j"},{"r","r"},{"l","l"},{"w","w"},{"v","w"},{"s","s"},{"h","h"},{"f","pʰ"},{"z","dz"},{"q","k"},{"x","tsʰ"}
    };
    static const std::unordered_map<std::string, std::string> keymap = {
        {"ʈ","t̪"},{"ʈʰ","t̪ʰ"},{"ɖ","d̪"},{"ɖʱ","d̪ʱ"},{"ɳ","n"},{"ŋ","n"},{"ɲ","n"},{"ʃ","s"},{"w","b"},
        {"ɽ","r"},{"ɽʱ","r"},{"ɔ","o"},{"ɛ","e"},{"z","dz"},{"f","pʰ"}
    };
    std::string clean;
    for (char c : r_in) if (c != '~') clean.push_back(c);
    auto toks = roman_tok(clean);
    std::string k;
    auto push = [&](const std::string & s) {
        if (!k.empty()) k += " ";
        k += s;
    };
    for (size_t idx = 0; idx < toks.size(); ++idx) {
        const auto & t = toks[idx];
        auto it = rc_ph.find(t);
        if (it != rc_ph.end()) {
            auto itK = keymap.find(it->second);
            push(itK != keymap.end() ? itK->second : it->second);
        } else if (t == "i" || t == "ii" || t == "ee") push("i");
        else if (t == "u" || t == "uu" || t == "oo") push("u");
        else if (t == "au") push("au");
        else if (t == "e") push("e");
        else if (t == "ai") push("ai");
        else if (t == "o") push("o");
        else if ((t == "a" || t == "aa") && (idx == 0 || idx == toks.size() - 1)) push("a");
    }
    return k;
}

inline std::string roman_rules(const std::string & r) {
    static const std::unordered_map<std::string, std::string> rc_deva = {
        {"chh","छ"},{"ch","च"},{"kh","ख"},{"gh","घ"},{"jh","झ"},{"th","थ"},{"dh","ध"},{"ph","फ"},{"bh","भ"},
        {"sh","श"},{"k","क"},{"g","ग"},{"c","क"},{"j","ज"},{"t","त"},{"d","द"},{"n","न"},{"p","प"},{"b","ब"},
        {"m","म"},{"y","य"},{"r","र"},{"l","ल"},{"w","व"},{"v","व"},{"s","स"},{"h","ह"},{"f","फ"},{"z","ज"},{"q","क"},{"x","छ"}
    };
    static const std::unordered_map<std::string, std::string> ind = {
        {"a","अ"},{"aa","आ"},{"i","इ"},{"ii","ई"},{"ee","ई"},{"u","उ"},{"uu","ऊ"},{"oo","ऊ"},{"e","ए"},{"ai","ऐ"},{"o","ओ"},{"au","औ"}
    };
    static const std::unordered_map<std::string, std::string> mat = {
        {"a",""},{"aa","ा"},{"i","ि"},{"ii","ी"},{"ee","ी"},{"u","ु"},{"uu","ू"},{"oo","ू"},{"e","े"},{"ai","ै"},{"o","ो"},{"au","ौ"}
    };
    auto toks = roman_tok(r);
    std::string out;
    bool prev_c = false;
    for (const auto & t : toks) {
        auto it = rc_deva.find(t);
        if (it != rc_deva.end()) {
            if (prev_c) out += "्";
            out += it->second;
            prev_c = true;
        } else {
            out += prev_c ? mat.at(t) : ind.at(t);
            prev_c = false;
        }
    }
    return out;
}

inline TranslitTables & get_translit_tables() {
    static TranslitTables tbl;
    if (!tbl.loaded) {
        tbl.loaded = true;
        tbl.direct = {
            {"gaun","गाउँ"},{"gau","गाउँ"},{"gaoon","गाउँ"},{"gaaun","गाउँ"},{"gaon","गाउँ"},
            {"gaunma","गाउँमा"},{"gauma","गाउँमा"},{"gaoonma","गाउँमा"},{"gaaunma","गाउँमा"},{"gaonma","गाउँमा"},
            {"gaunko","गाउँको"},{"gaunle","गाउँले"},{"gaupalika","गाउँपालिका"},{"gaaupaalika","गाउँपालिका"},
            {"thau","ठाउँ"},{"thaun","ठाउँ"},{"thaoon","ठाउँ"},{"thauma","ठाउँमा"},{"thaunma","ठाउँमा"},{"nau","नाउँ"},
            {"baschu","बस्छु"},{"baschhu","बस्छु"},{"auchu","आउँछु"},{"aauchu","आउँछु"},{"janchu","जान्छु"},
            {"khanchu","खान्छु"},{"garchu","गर्छु"},{"bhanchu","भन्छु"},{"huncha","हुन्छ"},{"hunchha","हुन्छ"},
            {"hudaina","हुँदैन"},{"chaina","छैन"},{"chhaina","छैन"},{"chau","छौ"},{"chhau","छौ"},{"chan","छन्"}
        };
        auto find_data_file = [](const std::string & name) -> std::string {
            for (const char * pfx : {"sakhiya/data/", "../sakhiya/data/", "/work/tts/repo/sakhiya/data/", "Sakhiya/sakhiya/data/"}) {
                std::string p = std::string(pfx) + name;
                std::ifstream f(p);
                if (f.good()) return p;
            }
            return "";
        };
        auto read_tsv = [&](const std::string & fname, std::unordered_map<std::string, std::string> & dst, bool first_wins = false) {
            std::string p = find_data_file(fname);
            if (p.empty()) return;
            std::ifstream f(p);
            std::string line;
            while (std::getline(f, line)) {
                auto tab = line.find('\t');
                if (tab != std::string::npos) {
                    std::string a = line.substr(0, tab);
                    auto tab2 = line.find('\t', tab + 1);
                    std::string b = line.substr(tab + 1, tab2 == std::string::npos ? std::string::npos : (tab2 - tab - 1));
                    if (!b.empty() && b.back() == '\r') b.pop_back();
                    if (!first_wins || !dst.count(a)) dst[a] = b;
                }
            }
        };
        read_tsv("lexkey_ne.tsv", tbl.ne, false);
        read_tsv("direct_ne.tsv", tbl.direct, false);
        // Re-apply hardcoded overrides from translit.py
        tbl.direct["gaun"] = "गाउँ"; tbl.direct["gau"] = "गाउँ"; tbl.direct["gaoon"] = "गाउँ"; tbl.direct["gaaun"] = "गाउँ"; tbl.direct["gaon"] = "गाउँ";
        tbl.direct["gaunma"] = "गाउँमा"; tbl.direct["gauma"] = "गाउँमा"; tbl.direct["gaoonma"] = "गाउँमा"; tbl.direct["gaaunma"] = "गाउँमा"; tbl.direct["gaonma"] = "गाउँमा";
        tbl.direct["thau"] = "ठाउँ"; tbl.direct["thaun"] = "ठाउँ"; tbl.direct["thaoon"] = "ठाउँ"; tbl.direct["thauma"] = "ठाउँमा"; tbl.direct["thaunma"] = "ठाउँमा";
        tbl.direct["baschu"] = "बस्छु"; tbl.direct["baschhu"] = "बस्छु";
        for (const char * td : {"thr_dangaura", "thr_rana", "thr_chitwania"}) {
            std::unordered_map<std::string, std::string> ex;
            read_tsv(std::string(td) + ".tsv", ex, true);
            tbl.thr_exact[td] = ex;
            for (const auto & kv : ex) {
                std::string rk = roman_key(kv.first);
                if (!rk.empty() && !tbl.thr_key[td].count(rk)) tbl.thr_key[td][rk] = kv.second;
            }
        }
    }
    return tbl;
}

inline std::string roman_to_deva(const std::string & word, const std::string & lang = "ne") {
    auto u = utf8_to_u32(word);
    for (char32_t c : u) {
        if (c >= 0x0900 && c <= 0x097F) return word;
    }
    bool has_tilde = (word.find('~') != std::string::npos);
    auto & tbl = get_translit_tables();
    std::string t_lang;
    if (lang == "thr" || lang == "thr_dangaura") t_lang = "thr_dangaura";
    else if (lang == "thr_rana") t_lang = "thr_rana";
    else if (lang == "thr_chitwania" || lang == "thr_chit") t_lang = "thr_chitwania";

    std::string w, w_lower;
    for (char c : word) {
        char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        w_lower.push_back(lc);
        if (lc >= 'a' && lc <= 'z') w.push_back(lc);
    }
    std::string k = roman_key(w);

    auto lookup = [&](const std::string & wk, const std::string & kk) -> std::string {
        if (!t_lang.empty()) {
            auto & ex = tbl.thr_exact[t_lang];
            if (ex.count(wk)) return ex[wk];
            auto & km = tbl.thr_key[t_lang];
            if (km.count(kk)) return km[kk];
        }
        if (tbl.direct.count(wk)) return tbl.direct[wk];
        if (!kk.empty() && tbl.ne.count(kk)) return tbl.ne[kk];
        return "";
    };

    static const std::vector<std::pair<std::string, std::string>> sfx_list = {
        {"haru","हरू"},{"sanga","सँग"},{"dekhi","देखि"},{"bata","बाट"},{"sana","सना"},
        {"lai","लाई"},{"ker","केर"},{"kar","कर"},{"hen","हेन"},{"tak","तक"},
        {"ma","मा"},{"me","मे"},{"ko","को"},{"ka","का"},{"ki","की"},{"le","ले"},{"ke","के"},{"se","से"}
    };

    auto split_fn = [&]() -> std::string {
        for (const auto & kv : sfx_list) {
            const auto & sf = kv.first;
            if (w.size() > sf.size() && w.size() - sf.size() >= 2 && str_ends_with(w, sf)) {
                std::string st = w.substr(0, w.size() - sf.size());
                std::string h = lookup(st, roman_key(st));
                if (!h.empty()) {
                    if (has_tilde && word.substr(0, word.size() - sf.size()).find('~') != std::string::npos && !str_ends_with(h, "ँ")) {
                        h += "ँ";
                    }
                    return h + kv.second;
                }
            }
        }
        return "";
    };

    std::string out;
    if (!t_lang.empty()) {
        auto & ex = tbl.thr_exact[t_lang];
        if (ex.count(w)) out = ex[w];
        else if (ex.count(w_lower)) out = ex[w_lower];
        if (out.empty()) out = split_fn();
        if (out.empty()) out = lookup(w, k);
    } else {
        out = lookup(w, k);
        if (out.empty()) out = split_fn();
    }
    if (out.empty()) out = roman_rules(w);
    if (has_tilde && out.find("ँ") == std::string::npos && out.find("ं") == std::string::npos) {
        out += "ँ";
    }
    return out;
}

inline std::string normalize_text(const std::string & text, const std::string & lang = "ne") {
    std::u32string u = nfc_normalize_u32(utf8_to_u32(text));
    std::u32string u_clean;
    for (char32_t c : u) {
        if (c != 0x200D && c != 0x200C) u_clean.push_back(c);
    }
    std::string t = expand_numbers(u32_to_utf8(u_clean));
    // Replace [A-Za-z][A-Za-z'~]* with roman_to_deva
    std::string out;
    size_t i = 0, n = t.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(t[i]);
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            size_t j = i + 1;
            while (j < n) {
                unsigned char cj = static_cast<unsigned char>(t[j]);
                if ((cj >= 'A' && cj <= 'Z') || (cj >= 'a' && cj <= 'z') || cj == '\'' || cj == '~') ++j;
                else break;
            }
            out += roman_to_deva(t.substr(i, j - i), lang);
            i = j;
        } else {
            out.push_back(t[i++]);
        }
    }
    return out;
}

// Forward declaration for Sanskrit
inline std::vector<std::string> sanskrit_text_to_phones(const std::string & text, bool add_end = true, int verse_opt = -1, bool keep_acc = false);
inline std::pair<std::vector<std::string>, std::vector<double>> sanskrit_text_to_phones_acc(const std::string & text, bool add_end = true, bool v2 = false);

inline std::vector<std::string> text_to_phones(const std::string & text, const std::string & lang = "ne", bool add_end = true) {
    if (lang == "sa") return sanskrit_text_to_phones(text, add_end);
    std::string t = normalize_text(text, lang);
    std::u32string u = utf8_to_u32(t);

    static const std::unordered_map<char32_t, std::string> punc_map = {
        {U',',","},{U'.',"."},{U'।',"."},{U'॥',"."},{U'?',"?"},{U'!',"!"},{U';',";"},{U':',","},{U'—',","},{U'-',","}
    };

    // Tokenize using _TOK = re.compile(r"[,.?!;।॥]|[\u0900-\u0963\u0966-\u097F]+|[A-Za-z][A-Za-z'~]*|\S")
    std::vector<std::u32string> toks;
    size_t i = 0, n = u.size();
    while (i < n) {
        char32_t c = u[i];
        if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == 0x00A0) {
            ++i;
            continue;
        }
        if (c == U',' || c == U'.' || c == U'?' || c == U'!' || c == U';' || c == U'।' || c == U'॥') {
            toks.push_back(std::u32string(1, c));
            ++i;
        } else if ((c >= 0x0900 && c <= 0x0963) || (c >= 0x0966 && c <= 0x097F)) {
            size_t j = i + 1;
            while (j < n && ((u[j] >= 0x0900 && u[j] <= 0x0963) || (u[j] >= 0x0966 && u[j] <= 0x097F))) ++j;
            toks.push_back(u.substr(i, j - i));
            i = j;
        } else if (is_ascii_alpha(c)) {
            size_t j = i + 1;
            while (j < n && (is_ascii_alpha(u[j]) || u[j] == U'\'' || u[j] == U'~')) ++j;
            toks.push_back(u.substr(i, j - i));
            i = j;
        } else {
            toks.push_back(std::u32string(1, c));
            ++i;
        }
    }

    std::vector<std::string> phones;
    for (size_t idx_t = 0; idx_t < toks.size(); ++idx_t) {
        const auto & m = toks[idx_t];
        if (m.size() == 1 && punc_map.count(m[0])) {
            if (!phones.empty() && phones.back() == "|") phones.pop_back();
            phones.push_back(punc_map.at(m[0]));
            phones.push_back("|");
            continue;
        }
        if (!(m[0] >= 0x0900 && m[0] <= 0x097F)) continue;
        std::string m_utf8 = u32_to_utf8(m);
        auto ph = word_to_phones(m_utf8);
        if (!ph.empty()) {
            for (size_t idx_p = 0; idx_p < ph.size(); ++idx_p) {
                if (!(ph[idx_p] == "h" && idx_p > 0 && ph[idx_p - 1] == "h")) {
                    phones.push_back(ph[idx_p]);
                }
            }
            phones.push_back("|");
            std::string nxt_w = (idx_t + 1 < toks.size()) ? u32_to_utf8(toks[idx_t + 1]) : "";
            if ((str_ends_with(m_utf8, "नुहोस्") || str_ends_with(m_utf8, "नोस्")) &&
                (nxt_w == "र" || nxt_w == "अनि" || nxt_w == "तर" || nxt_w == "वा")) {
                phones.pop_back();
                phones.push_back(",");
                phones.push_back("|");
            }
        }
    }
    if (!phones.empty() && phones.back() == "|") phones.pop_back();
    if (add_end && (phones.empty() || (phones.back() != "." && phones.back() != "?" && phones.back() != "!"))) {
        phones.push_back(".");
    }
    return phones;
}

inline std::vector<double> phone_dm(const std::vector<std::string> & ph) {
    size_t n = ph.size();
    std::vector<double> dm(n, 1.0);
    for (size_t i = 0; i < n; ++i) {
        const auto & p = ph[i];
        if (p == ",") {
            dm[i] = 1.25;
        } else if (p == "!" || p == "?") {
            dm[i] = 1.32;
        } else if (p == "a" && i >= 1 && ph[i - 1] == "w" && i + 2 < n && ph[i + 1] == "h" &&
                   (ph[i + 2] == "!" || ph[i + 2] == "|")) {
            dm[i] = 1.38;
            dm[i + 1] = 0.72;
        }
    }
    return dm;
}

inline std::pair<std::vector<std::string>, std::vector<double>> text_to_phones_acc(
    const std::string & text, const std::string & lang = "ne", bool add_end = true) {
    if (lang == "sa") return sanskrit_text_to_phones_acc(text, add_end);
    auto ph = text_to_phones(text, lang, add_end);
    size_t n = ph.size();
    std::vector<double> off(n, 0.0);
    size_t c_start = 0;
    const double pi = std::acos(-1.0);
    for (size_t i = 0; i < n; ++i) {
        const auto & p = ph[i];
        if (p == "," || p == "." || p == "?" || p == "!" || p == ";") {
            std::vector<size_t> vs;
            for (size_t j = c_start; j < i; ++j) {
                if (starts_with_vowel_base(ph[j])) vs.push_back(j);
            }
            size_t nv = vs.size();
            if (p == "!") {
                for (size_t k = 0; k < nv; ++k) {
                    off[vs[k]] = py_round4(0.95 * std::sin(pi * (static_cast<double>(k) + 0.5) / std::max<size_t>(nv, 1)) + 0.30);
                }
                if (nv == 1) {
                    off[vs[0]] = 1.35;
                } else if (nv >= 2) {
                    off[vs[0]] = std::max(off[vs[0]], 1.15);
                    off[vs.back()] = 0.55;
                }
            } else if (p == "?") {
                for (size_t k = 0; k < nv; ++k) {
                    double denom = static_cast<double>(nv > 1 ? (nv - 1) : 1);
                    off[vs[k]] = py_round4(-0.20 + 0.35 * (static_cast<double>(k) / denom));
                }
                if (nv >= 1) off[vs.back()] = 1.25;
                if (nv >= 2) off[vs[nv - 2]] = 0.68;
            } else if (p == ",") {
                if (nv >= 1) off[vs.back()] = 0.24;
                if (nv >= 2) off[vs[nv - 2]] = 0.12;
            }
            c_start = i + 1;
        }
    }
    std::vector<double> sm = off;
    for (size_t i = 0; i < n; ++i) {
        const auto & p = ph[i];
        if (!is_punct_phone(p) && !starts_with_vowel_base(p)) {
            double lv = (i > 0) ? off[i - 1] : 0.0;
            double rv = (i + 1 < n) ? off[i + 1] : 0.0;
            sm[i] = py_round4(0.5 * (lv + rv));
        }
    }
    return {ph, sm};
}

// ---------------- Sanskrit (sakhiya/sanskrit.py) ----------------
inline bool is_sa_vowel_phone(const std::string & p) {
    if (p.empty() || utf8_to_u32(p).size() > 2) return false;
    return starts_with_vowel_base(p);
}

inline std::string iast_to_deva(const std::string & text) {
    std::u32string u = nfc_normalize_u32(utf8_to_u32(text));
    // Intervocalic ḷ -> ḻ
    auto is_iast_v = [](char32_t c) {
        switch (std::towlower(c)) {
            case U'a': case U'e': case U'i': case U'o': case U'u':
            case 0x0101: case 0x012B: case 0x016B: case 0x1E5B: case 0x1E5D: return true;
            default: return false;
        }
    };
    for (size_t i = 1; i + 1 < u.size(); ++i) {
        if ((u[i] == 0x1E37 || u[i] == 0x1E36) && is_iast_v(u[i - 1]) && is_iast_v(u[i + 1])) {
            u[i] = 0x1E3B; // ḻ
        }
    }
    bool has_diacritics = false;
    static const std::u32string diac_chars = U"āīūṛṝḷḻṅñṭḍṇśṣṃṁḥĀĪŪṚṜḶḺṄÑṬḌṆŚṢṂṀḤ";
    for (char32_t c : u) {
        if (diac_chars.find(c) != std::u32string::npos) {
            has_diacritics = true;
            break;
        }
    }
    std::string t = u32_to_utf8(u);
    if (!has_diacritics) {
        replace_all(t, "aa", "ā");
        replace_all(t, "ii", "ī");
        replace_all(t, "uu", "ū");
        replace_all(t, "sh", "ś");
    }
    u = utf8_to_u32(t);

    static const std::vector<std::pair<std::u32string, std::string>> tok_list = {
        {U"au","V:औ:ौ"},{U"ai","V:ऐ:ै"},
        {U"kh","C:ख"},{U"gh","C:घ"},{U"ch","C:छ"},{U"jh","C:झ"},{U"ṭh","C:ठ"},{U"ḍh","C:ढ"},
        {U"th","C:थ"},{U"dh","C:ध"},{U"ph","C:फ"},{U"bh","C:भ"},
        {U"a","V:अ:"},{U"ā","V:आ:ा"},{U"i","V:इ:ि"},{U"ī","V:ई:ी"},{U"u","V:उ:ु"},{U"ū","V:ऊ:ू"},
        {U"ṛ","V:ऋ:ृ"},{U"ṝ","V:ॠ:ॄ"},{U"ḷ","V:ऌ:ॢ"},{U"e","V:ए:े"},{U"o","V:ओ:ो"},
        {U"k","C:क"},{U"g","C:ग"},{U"ṅ","C:ङ"},{U"c","C:च"},{U"j","C:ज"},{U"ñ","C:ञ"},
        {U"ṭ","C:ट"},{U"ḍ","C:ड"},{U"ṇ","C:ण"},{U"t","C:त"},{U"d","C:द"},{U"n","C:न"},
        {U"p","C:प"},{U"b","C:ब"},{U"m","C:म"},{U"y","C:य"},{U"r","C:र"},{U"l","C:ल"},
        {U"ḻ","C:ळ"},{U"v","C:व"},{U"ś","C:श"},{U"ṣ","C:ष"},{U"s","C:स"},{U"h","C:ह"},
        {U"ṃ","O:ं"},{U"ṁ","O:ं"},{U"ḥ","O:ः"},{U"'","O:ऽ"},{U"’","O:ऽ"}
    };

    auto is_iast_word_cp = [](char32_t c) {
        if (is_ascii_alpha(c) || c == U'\'' || c == U'’') return true;
        return diac_chars.find(c) != std::u32string::npos;
    };

    auto convert_word = [&](const std::u32string & w_in) -> std::string {
        std::u32string w;
        for (char32_t c : w_in) {
            if (c >= U'A' && c <= U'Z') w.push_back(c - U'A' + U'a');
            else w.push_back(c);
        }
        std::string out;
        bool pc = false;
        size_t i = 0;
        while (i < w.size()) {
            bool matched = false;
            for (const auto & kv : tok_list) {
                const auto & pat = kv.first;
                if (i + pat.size() <= w.size() && w.compare(i, pat.size(), pat) == 0) {
                    const std::string & spec = kv.second;
                    if (spec[0] == 'C') {
                        if (pc) out += "्";
                        out += spec.substr(2);
                        pc = true;
                    } else if (spec[0] == 'V') {
                        auto colon = spec.find(':', 2);
                        std::string iv_s = spec.substr(2, colon - 2);
                        std::string mt_s = spec.substr(colon + 1);
                        out += pc ? mt_s : iv_s;
                        pc = false;
                    } else if (spec[0] == 'O') {
                        out += spec.substr(2);
                        pc = false;
                    }
                    i += pat.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) ++i;
        }
        return out;
    };

    std::string res;
    size_t i = 0;
    while (i < u.size()) {
        if (is_iast_word_cp(u[i])) {
            size_t j = i + 1;
            while (j < u.size() && is_iast_word_cp(u[j])) ++j;
            res += convert_word(u.substr(i, j - i));
            i = j;
        } else {
            append_u32_to_utf8(u[i++], res);
        }
    }
    return res;
}

inline std::string replace_om_re(const std::string & s_in) {
    // _OM_RE = re.compile(r"ओ(?:[३3]म्?|म्|ं(?![क-ह]))")
    std::u32string u = utf8_to_u32(s_in);
    std::u32string out;
    size_t i = 0, n = u.size();
    while (i < n) {
        if (u[i] == U'ओ' && i + 1 < n) {
            if (u[i + 1] == U'३' || u[i + 1] == U'3') {
                size_t adv = 2;
                if (i + 2 < n && u[i + 2] == U'म') {
                    adv = 3;
                    if (i + 3 < n && u[i + 3] == 0x094D) adv = 4;
                }
                out.push_back(U'ॐ');
                i += adv;
                continue;
            }
            if (u[i + 1] == U'म' && i + 2 < n && u[i + 2] == 0x094D) {
                out.push_back(U'ॐ');
                i += 3;
                continue;
            }
            if (u[i + 1] == 0x0902) { // ं
                bool next_cons = (i + 2 < n && u[i + 2] >= 0x0915 && u[i + 2] <= 0x0939);
                if (!next_cons) {
                    out.push_back(U'ॐ');
                    i += 2;
                    continue;
                }
            }
        }
        out.push_back(u[i++]);
    }
    return u32_to_utf8(out);
}

inline std::string restore_rv_svara(std::string t) {
    if (t.find("अग्निमीळे") != std::string::npos && t.find("पुरोहित") != std::string::npos &&
        t.find("\xE0\xA5\x91") == std::string::npos) {
        replace_all(t, "अग्निमीळे", "अ॒ग्निमी॑ळे");
        replace_all(t, "पुरोहितं", "पु॒रोहि॑तं");
        replace_all(t, "यज्ञस्य", "य॒ज्ञस्य॑");
        replace_all(t, "देवमृत्विजम्", "दे॒वमृ॒त्विज॑म्");
        replace_all(t, "होतारं", "होता॑रं");
        replace_all(t, "रत्नधातमम्", "रत्न॒धात॑मम्");
    }
    return t;
}

inline std::vector<std::string> sanskrit_word_phones(const std::string & w_in, bool keep_acc = false) {
    std::u32string u_raw = nfc_normalize_u32(utf8_to_u32(w_in));
    std::u32string u_filt;
    for (char32_t c : u_raw) {
        if ((c >= 0x0953 && c <= 0x0954) || (c >= 0x1CD0 && c <= 0x1CFF)) continue;
        if (!keep_acc && (c == 0x0951 || c == 0x0952)) continue;
        u_filt.push_back(c);
    }
    std::string w_str = replace_om_re(u32_to_utf8(u_filt));
    replace_all(w_str, "ज्ञ", "ग्ञ");
    std::u32string w = utf8_to_u32(w_str);

    const auto & C = get_deva_c();
    const auto & NUKTA = get_nukta_map();
    const auto & NP = get_nasal_place();
    static const std::unordered_map<char32_t, std::vector<std::string>> long_v = {
        {U'आ',{"a"}},{U'ा',{"a"}},{U'ई',{"i"}},{U'ी',{"i"}},{U'ऊ',{"u"}},{U'ू',{"u"}},{U'ॄ',{"r","i"}},{U'ॠ',{"r","i"}}
    };
    static const std::unordered_map<char32_t, std::string> short_v = {
        {U'अ',"ʌ"},{U'इ',"i"},{U'उ',"u"},{U'ए',"e"},{U'ओ',"o"},{U'ि',"i"},{U'ु',"u"},{U'े',"e"},{U'ो',"o"}
    };
    static const std::unordered_map<char32_t, std::vector<std::string>> diph = {
        {U'ऐ',{"ʌ","i"}},{U'ै',{"ʌ","i"}},{U'औ',{"ʌ","u"}},{U'ौ',{"ʌ","u"}},
        {U'ऋ',{"r","i"}},{U'ृ',{"r","i"}},{U'ऌ',{"l","r","i"}},{U'ॢ',{"l","r","i"}}
    };

    std::vector<std::string> out;
    auto vowel_before = [&]() -> std::string {
        for (int k = static_cast<int>(out.size()) - 1; k >= 0; --k) {
            const auto & p = out[k];
            if (p == "ʌ" || p == "a" || p == "i" || p == "u" || p == "e" || p == "o") return p;
        }
        return "ʌ";
    };
    auto nasalize_last = [&]() {
        for (int k = static_cast<int>(out.size()) - 1; k >= 0; --k) {
            const auto & p = out[k];
            if (p == "ʌ" || p == "a" || p == "i" || p == "u" || p == "e" || p == "o") {
                out[k] += "\xCC\x83";
                break;
            }
        }
    };

    size_t i = 0, n = w.size();
    while (i < n) {
        char32_t ch = w[i];
        if (ch == U'ॐ') {
            out.push_back("o"); out.push_back("m"); ++i;
        } else if (C.count(ch)) {
            std::string ph = C.at(ch);
            if (i + 1 < n && w[i + 1] == 0x093C) {
                if (NUKTA.count(ch)) ph = NUKTA.at(ch);
                ++i;
            }
            out.push_back(ph);
            ++i;
            char32_t nx = (i < n) ? w[i] : 0;
            if (nx == 0x094D) {
                ++i;
            } else if (long_v.count(nx)) {
                for (const auto & p : long_v.at(nx)) out.push_back(p);
                ++i;
            } else if (short_v.count(nx)) {
                out.push_back(short_v.at(nx));
                ++i;
            } else if (diph.count(nx)) {
                for (const auto & p : diph.at(nx)) out.push_back(p);
                ++i;
            } else {
                out.push_back("ʌ");
            }
        } else if (long_v.count(ch)) {
            for (const auto & p : long_v.at(ch)) out.push_back(p);
            ++i;
        } else if (short_v.count(ch)) {
            out.push_back(short_v.at(ch));
            ++i;
        } else if (diph.count(ch)) {
            for (const auto & p : diph.at(ch)) out.push_back(p);
            ++i;
        } else if (ch == 0x0901) { // CHAN ँ
            nasalize_last(); ++i;
        } else if (ch == 0x0902) { // ANU ं
            char32_t nx = (i + 1 < n) ? w[i + 1] : 0;
            if (NP.count(nx)) out.push_back(NP.at(nx));
            else if (nx == 0) out.push_back("m");
            else nasalize_last();
            ++i;
        } else if (ch == 0x0903) { // VIS ः
            if (out.empty() || out.back() != "h") out.push_back("h");
            ++i;
        } else if (ch == U'ऽ') {
            out.push_back(vowel_before()); ++i;
        } else if (keep_acc && (ch == 0x0951 || ch == 0x0952)) {
            out.push_back(cp_to_utf8(ch)); ++i;
        } else {
            ++i;
        }
    }
    std::vector<std::string> clean;
    for (size_t k = 0; k < out.size(); ++k) {
        if (!(out[k] == "h" && k > 0 && out[k - 1] == "h")) clean.push_back(out[k]);
    }
    return clean;
}

inline std::vector<std::string> syllable_weights(const std::string & w_in) {
    std::u32string u_raw = nfc_normalize_u32(utf8_to_u32(w_in));
    std::u32string u_filt;
    for (char32_t c : u_raw) {
        if ((c >= 0x0951 && c <= 0x0954) || (c >= 0x1CD0 && c <= 0x1CFF)) continue;
        u_filt.push_back(c);
    }
    std::string w_str = replace_om_re(u32_to_utf8(u_filt));
    replace_all(w_str, "ॐ", "");
    replace_all(w_str, "ज्ञ", "ग्ञ");
    std::u32string w = utf8_to_u32(w_str);

    const auto & C = get_deva_c();
    static const std::u32string long_orth = U"आाईीऊूएेऐैओोऔौॠॄ";
    static const std::u32string all_v = U"आाईीऊूएेऐैओोऔौॠॄअइउिुृॢऋऌ";
    auto is_long = [&](char32_t c) { return long_orth.find(c) != std::u32string::npos; };
    auto is_v = [&](char32_t c) { return all_v.find(c) != std::u32string::npos; };

    struct SylInfo { bool lg; int c; int va; };
    std::vector<SylInfo> syls;
    size_t i = 0, n = w.size();
    while (i < n) {
        char32_t ch = w[i];
        if (C.count(ch)) {
            if (i + 1 < n && w[i + 1] == 0x093C) ++i;
            if (!syls.empty() && syls.back().va == 0) syls.back().c += 1;
            ++i;
            char32_t nx = (i < n) ? w[i] : 0;
            if (nx == 0x094D) {
                ++i;
            } else if (is_v(nx)) {
                syls.push_back({is_long(nx), 0, 0});
                ++i;
            } else {
                syls.push_back({false, 0, 0});
            }
        } else if (is_v(ch)) {
            syls.push_back({is_long(ch), 0, 0});
            ++i;
        } else if (ch == 0x0901 || ch == 0x0902 || ch == 0x0903) {
            if (!syls.empty()) syls.back().va = 1;
            ++i;
        } else {
            ++i;
        }
    }
    std::vector<std::string> res;
    for (const auto & s : syls) {
        res.push_back((s.lg || s.c >= 2 || s.va != 0) ? "G" : "L");
    }
    return res;
}

inline std::vector<std::string> sanskrit_text_to_phones(
    const std::string & text, bool add_end, int verse_opt, bool keep_acc) {
    std::string t = iast_to_deva(nfc_normalize(text));
    t = replace_om_re(t);
    t = restore_rv_svara(t);
    replace_all(t, "\xE2\x80\x8D", "");
    replace_all(t, "\xE2\x80\x8C", "");

    // Split by (?<=[।॥])|\n+
    std::u32string u = utf8_to_u32(t);
    std::vector<std::u32string> segs;
    std::u32string cur_seg;
    for (size_t i = 0; i < u.size(); ++i) {
        char32_t c = u[i];
        if (c == U'\n') {
            if (!trim_str(u32_to_utf8(cur_seg)).empty()) segs.push_back(cur_seg);
            cur_seg.clear();
        } else {
            cur_seg.push_back(c);
            if (c == U'।' || c == U'॥') {
                if (!trim_str(u32_to_utf8(cur_seg)).empty()) segs.push_back(cur_seg);
                cur_seg.clear();
            }
        }
    }
    if (!trim_str(u32_to_utf8(cur_seg)).empty()) segs.push_back(cur_seg);

    bool verse = (verse_opt >= 0) ? (verse_opt != 0) : (t.find("॥") != std::string::npos || segs.size() >= 2);
    std::vector<std::string> ph;

    for (const auto & seg32 : segs) {
        // toks = re.findall(r"[\u0900-\u097F]+", re.sub("[।॥]", " ", seg))
        std::vector<std::u32string> toks;
        size_t i = 0;
        while (i < seg32.size()) {
            char32_t c = seg32[i];
            if (c >= 0x0900 && c <= 0x097F && c != U'।' && c != U'॥') {
                size_t j = i + 1;
                while (j < seg32.size() && seg32[j] >= 0x0900 && seg32[j] <= 0x097F &&
                       seg32[j] != U'।' && seg32[j] != U'॥') ++j;
                toks.push_back(seg32.substr(i, j - i));
                i = j;
            } else {
                ++i;
            }
        }
        std::vector<std::string> words;
        std::map<size_t, int> om_before;
        for (const auto & w32 : toks) {
            int om_cnt = 0;
            std::u32string w_clean;
            for (char32_t c : w32) {
                if (c == U'ॐ') ++om_cnt;
                else w_clean.push_back(c);
            }
            if (om_cnt > 0) om_before[words.size()] += om_cnt;
            if (!w_clean.empty()) words.push_back(u32_to_utf8(w_clean));
        }
        int om_tail = 0;
        if (om_before.count(words.size())) {
            om_tail = om_before[words.size()];
            om_before.erase(words.size());
        }
        std::vector<int> sylls;
        int tot = 0;
        for (const auto & w : words) {
            int sc = static_cast<int>(syllable_weights(w).size());
            sylls.push_back(sc);
            tot += sc;
        }
        std::map<size_t, std::string> marks;
        std::vector<std::pair<int, std::string>> plan;
        if (tot >= 28 && tot <= 36) plan = {{8, ","}, {16, ";"}, {24, ","}};
        else if (tot >= 14 && tot <= 18) plan = {{8, ","}};
        if (verse && !plan.empty() && words.size() > 1) {
            std::vector<int> cum(words.size(), 0);
            int s_acc = 0;
            for (size_t k = 0; k < words.size(); ++k) {
                s_acc += sylls[k];
                cum[k] = s_acc;
            }
            for (const auto & qp : plan) {
                int q = qp.first;
                size_t best_k = 0;
                for (size_t k = 1; k + 1 < words.size(); ++k) {
                    if (std::abs(cum[k] - q) < std::abs(cum[best_k] - q)) best_k = k;
                }
                if (std::abs(cum[best_k] - q) <= 3) {
                    marks[best_k] = (marks.count(best_k) && marks[best_k] == ";") ? ";" : qp.second;
                }
            }
        }
        for (size_t k = 0; k < words.size(); ++k) {
            if (om_before.count(k)) {
                ph.push_back("<om>");
                ph.push_back(",");
                ph.push_back("|");
            }
            auto p = sanskrit_word_phones(words[k], keep_acc);
            if (!p.empty()) {
                ph.insert(ph.end(), p.begin(), p.end());
                ph.push_back("|");
            }
            if (marks.count(k)) {
                if (!ph.empty() && ph.back() == "|") ph.pop_back();
                ph.push_back(marks[k]);
                ph.push_back("|");
            }
        }
        std::string end_tok = (seg32.find(U'॥') != std::u32string::npos || !verse) ? "." : ";";
        for (char32_t c : seg32) {
            if (c == U'?' || c == U'!') end_tok = (c == U'?') ? "?" : "!";
        }
        if (!ph.empty() && ph.back() == "|") ph.pop_back();
        if (om_tail > 0) {
            if (words.empty()) {
                ph.push_back("<om>");
            } else {
                ph.push_back(",");
                ph.push_back("<om>");
            }
        }
        if (!words.empty() || om_tail > 0) {
            ph.push_back(end_tok);
            ph.push_back("|");
        }
    }
    if (!ph.empty() && ph.back() == "|") ph.pop_back();
    if (!ph.empty() && ph.back() == ";") ph.back() = ".";
    if (add_end && (ph.empty() || (ph.back() != "." && ph.back() != "?" && ph.back() != "!"))) {
        ph.push_back(".");
    }
    return ph;
}

inline std::pair<std::vector<std::string>, std::vector<double>> svara_offsets(const std::vector<std::string> & ph) {
    const std::string m_udatta = "\xE0\xA5\x91";  // U+0951
    const std::string m_anudatta = "\xE0\xA5\x92"; // U+0952
    static const std::unordered_map<std::string, double> cadence = {
        {",", 0.5}, {";", 0.3}, {".", -0.7}, {"?", 0.8}, {"!", -0.2}
    };

    std::vector<std::string> out;
    std::vector<double> off;
    bool has_marks = false;
    for (const auto & p : ph) {
        if (p == m_udatta || p == m_anudatta) {
            has_marks = true;
            int j = static_cast<int>(out.size()) - 1;
            while (j >= 0 && !is_sa_vowel_phone(out[j])) --j;
            if (j >= 0) {
                double v = (p == m_udatta) ? 1.30 : -1.75;
                off[j] = v;
                if (j > 0 && out[j - 1] == out[j] && out[j] != "ʌ") off[j - 1] = v;
            }
            continue;
        }
        out.push_back(p);
        off.push_back(0.0);
    }

    bool is_shloka = false;
    for (const auto & p : out) if (p == ";") is_shloka = true;
    for (const auto & p : ph) if (p == "<om>") is_shloka = true;

    if (!has_marks && !is_shloka) return {out, off};

    const double pi = std::acos(-1.0);
    if (!has_marks && is_shloka) {
        size_t p_st = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            if (cadence.count(out[i])) {
                std::vector<size_t> vs;
                for (size_t j = p_st; j < i; ++j) {
                    if (is_sa_vowel_phone(out[j])) vs.push_back(j);
                }
                size_t nv = vs.size();
                for (size_t k = 0; k < nv; ++k) {
                    size_t j = vs[k];
                    double arch = 0.38 * std::sin(pi * (static_cast<double>(k) + 0.5) / std::max<size_t>(nv, 1)) - 0.10;
                    int nxt_c = 0;
                    for (size_t m = j + 1; m < std::min(out.size(), j + 4); ++m) {
                        if (!is_sa_vowel_phone(out[m]) && out[m] != "|" && !cadence.count(out[m]) && out[m] != "<om>") {
                            ++nxt_c;
                        }
                    }
                    bool guru = (out[j] == "a" || out[j] == "e" || out[j] == "o" || nxt_c >= 2);
                    off[j] = py_round4(arch + (guru ? 0.16 : -0.12));
                }
                p_st = i + 1;
            }
        }
    }

    int st = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const auto & p = out[i];
        if (p == "|" || cadence.count(p)) {
            st = 0;
        } else if (is_sa_vowel_phone(p)) {
            if (off[i] > 0.5) st = 1;
            else if (off[i] < -0.5) st = 0;
            else if (st == 1 && off[i] == 0.0) off[i] = 0.35;
        }
    }
    for (size_t i = 0; i < out.size(); ++i) {
        auto it = cadence.find(out[i]);
        if (it != cadence.end()) {
            std::vector<int> js;
            int min_j = std::max(-1, static_cast<int>(i) - 7);
            for (int j = static_cast<int>(i) - 1; j > min_j && js.size() < 2; --j) {
                if (is_sa_vowel_phone(out[j])) js.push_back(j);
            }
            for (size_t n_idx = 0; n_idx < js.size(); ++n_idx) {
                off[js[n_idx]] += it->second * (n_idx == 0 ? 1.0 : 0.4) * (has_marks ? 1.0 : 0.65);
            }
        }
    }
    std::vector<double> sm = off;
    for (size_t i = 0; i < out.size(); ++i) {
        const auto & p = out[i];
        if (!is_sa_vowel_phone(p) && p != "|" && !cadence.count(p)) {
            double lv = (i > 0) ? off[i - 1] : 0.0;
            double rv = (i + 1 < out.size()) ? off[i + 1] : 0.0;
            if (lv < -0.8 || rv < -0.8) sm[i] = -0.72;
            else if (lv > 0.8 || rv > 0.8) sm[i] = 0.72;
            else sm[i] = 0.5 * (lv + rv);
        }
    }
    return {out, sm};
}

struct LayaResult {
    std::vector<std::string> ph;
    std::vector<double> off;
    std::vector<double> dm;
};

inline LayaResult expand_om(const std::vector<std::string> & ph,
                            const std::vector<double> & off,
                            const std::vector<double> * dm_opt,
                            bool v2) {
    LayaResult r;
    for (size_t i = 0; i < ph.size(); ++i) {
        if (ph[i] == "<om>") {
            if (v2) {
                r.ph.push_back("o"); r.ph.push_back("m");
                r.off.push_back(0.28); r.off.push_back(0.14);
                r.dm.push_back(1.55); r.dm.push_back(1.30);
            } else {
                r.ph.push_back("o"); r.ph.push_back("o"); r.ph.push_back("m");
                r.off.push_back(0.35); r.off.push_back(0.65); r.off.push_back(0.20);
                r.dm.push_back(1.55); r.dm.push_back(2.05); r.dm.push_back(1.50);
            }
        } else {
            r.ph.push_back(ph[i]);
            r.off.push_back(off[i]);
            r.dm.push_back(dm_opt ? (*dm_opt)[i] : 1.0);
        }
    }
    return r;
}

inline std::pair<std::vector<std::string>, std::vector<double>> sanskrit_text_to_phones_acc(
    const std::string & text, bool add_end, bool v2) {
    auto so = svara_offsets(sanskrit_text_to_phones(text, add_end, -1, true));
    auto ex = expand_om(so.first, so.second, nullptr, v2);
    return {ex.ph, ex.off};
}

inline LayaResult laya_scales(const std::string & text, bool add_end = true, bool v2 = false) {
    std::string t = restore_rv_svara(replace_om_re(iast_to_deva(nfc_normalize(text))));
    replace_all(t, "\xE2\x80\x8D", "");
    replace_all(t, "\xE2\x80\x8C", "");
    std::u32string u = utf8_to_u32(t);
    std::vector<std::string> w_syls;
    size_t idx = 0;
    while (idx < u.size()) {
        char32_t c = u[idx];
        if (c >= 0x0900 && c <= 0x097F && c != U'।' && c != U'॥') {
            size_t j = idx + 1;
            while (j < u.size() && u[j] >= 0x0900 && u[j] <= 0x097F && u[j] != U'।' && u[j] != U'॥') ++j;
            std::u32string w_clean;
            for (size_t k = idx; k < j; ++k) if (u[k] != U'ॐ') w_clean.push_back(u[k]);
            if (!w_clean.empty()) {
                for (const auto & sw : syllable_weights(u32_to_utf8(w_clean))) w_syls.push_back(sw);
            }
            idx = j;
        } else {
            ++idx;
        }
    }

    auto raw_ph = sanskrit_text_to_phones(text, add_end, -1, true);
    bool has_marks = false;
    bool has_semi_or_om = false;
    for (const auto & p : raw_ph) {
        if (p == "\xE0\xA5\x91" || p == "\xE0\xA5\x92") has_marks = true;
        if (p == ";" || p == "<om>") has_semi_or_om = true;
    }
    bool is_shloka = (!has_marks) && has_semi_or_om;
    auto so = svara_offsets(raw_ph);
    const auto & ph = so.first;
    const auto & off = so.second;
    size_t n = ph.size();
    std::vector<double> dm(n, 1.0);

    if (v2) {
        size_t si_v2 = 0, i_v2 = 0;
        while (i_v2 < n) {
            const auto & p = ph[i_v2];
            if (is_sa_vowel_phone(p)) {
                bool diph = (p == "ʌ" && i_v2 + 1 < n && (ph[i_v2 + 1] == "i" || ph[i_v2 + 1] == "u"));
                bool g = (si_v2 < w_syls.size())
                             ? (w_syls[si_v2] == "G")
                             : (p == "a" || p == "e" || p == "o" || diph || utf8_to_u32(p).size() > 1);
                ++si_v2;
                if (diph) {
                    dm[i_v2] = 1.02; dm[i_v2 + 1] = 1.02; i_v2 += 2;
                } else {
                    dm[i_v2] = g ? 1.04 : 0.97; i_v2 += 1;
                }
            } else {
                if (p == "," || p == ";") dm[i_v2] = 1.12;
                ++i_v2;
            }
        }
        return expand_om(ph, off, &dm, true);
    }

    size_t si = 0, i = 0;
    while (i < n) {
        const auto & p = ph[i];
        if (is_sa_vowel_phone(p)) {
            bool diph = (p == "ʌ" && i + 1 < n && (ph[i + 1] == "i" || ph[i + 1] == "u"));
            bool g = (si < w_syls.size())
                         ? (w_syls[si] == "G")
                         : (p == "a" || p == "e" || p == "o" || diph || utf8_to_u32(p).size() > 1);
            ++si;
            if (diph) {
                double d_val = is_shloka ? 1.08 : 1.04;
                dm[i] = d_val; dm[i + 1] = d_val; i += 2;
            } else {
                dm[i] = is_shloka ? (g ? 1.20 : 0.90) : (g ? 1.08 : 0.98);
                if (std::abs(off[i]) >= 0.8) dm[i] = std::max(dm[i], 1.35);
                i += 1;
            }
        } else {
            if (p == "h" && (i + 1 == n || ph[i + 1] == "|" || ph[i + 1] == "," || ph[i + 1] == ";" ||
                             ph[i + 1] == "." || ph[i + 1] == "?" || ph[i + 1] == "!")) {
                dm[i] = is_shloka ? 1.18 : 1.12;
            }
            ++i;
        }
    }
    return expand_om(ph, off, &dm, false);
}

// ---------------- Chandas-Laya-Svara (sakhiya/chandas.py) ----------------
struct AksharaInfo {
    std::string ak;
    bool is_long = false;
    std::string svara;
    bool has_yoga = false;
    int onset_hal = 0;
    int trailing_c = 0;
    std::string weight;
    double beats = 1.0;
    double f0_cents = 0.0;
    double target_ms = 0.0;
};

struct ChandasResult {
    std::string meter;
    bool has_vedic_marks = false;
    double bpm = 75.0;
    double tonic_hz = 215.0;
    std::vector<AksharaInfo> aksharas;
    std::vector<std::string> ph;
    std::vector<double> off;
    std::vector<double> dm;
    std::vector<int> dur_frames;
    std::vector<double> f0_cents;
};

inline std::vector<AksharaInfo> scan_word_aksharas(const std::string & word) {
    std::u32string w_raw = nfc_normalize_u32(utf8_to_u32(word));
    std::string w_str = replace_om_re(u32_to_utf8(w_raw));
    replace_all(w_str, "ज्ञ", "ग्ञ");
    std::u32string w_tmp = utf8_to_u32(w_str);
    std::u32string w;
    for (char32_t c : w_tmp) {
        if ((c >= 0x0953 && c <= 0x0954) || (c >= 0x1CD0 && c <= 0x1CFF)) continue;
        w.push_back(c);
    }
    if (w == U"ॐ") {
        AksharaInfo a;
        a.ak = "ॐ"; a.weight = "G"; a.beats = 2.0; a.svara = "om";
        a.is_long = true; a.has_yoga = true; a.trailing_c = 1;
        return {a};
    }

    const auto & C = get_deva_c();
    static const std::u32string long_v_chars = U"आाईीऊूॠॄएेऐैओोऔौ";
    static const std::u32string matra_chars = U"ािीुूृॄॢेैोौ";
    static const std::u32string indep_v_chars = U"अआइईउऊऋॠऌएऐओऔ";
    static const std::u32string yoga_chars = U"ंँः";
    auto is_long_v = [&](char32_t c) { return long_v_chars.find(c) != std::u32string::npos; };
    auto is_matra = [&](char32_t c) { return matra_chars.find(c) != std::u32string::npos; };
    auto is_indep = [&](char32_t c) { return indep_v_chars.find(c) != std::u32string::npos; };
    auto is_yoga = [&](char32_t c) { return yoga_chars.find(c) != std::u32string::npos; };

    struct RawUnit {
        std::string type;
        std::u32string text;
        bool is_long = false;
        std::string svara = "udatta";
        bool yoga = false;
    };
    std::vector<RawUnit> units;
    size_t i = 0, n = w.size();
    while (i < n) {
        char32_t ch = w[i];
        if (ch == U'ॐ') {
            units.push_back({"OM", U"ॐ", true, "om", true});
            ++i;
            continue;
        }
        if (C.count(ch)) {
            std::u32string c_txt(1, ch);
            ++i;
            if (i < n && w[i] == 0x093C) c_txt.push_back(w[i++]);
            if (i < n && w[i] == 0x094D) {
                c_txt.push_back(0x094D);
                ++i;
                units.push_back({"C_HAL", c_txt, false, "udatta", false});
                continue;
            }
            std::u32string v_txt;
            bool is_lg = false;
            if (i < n && is_matra(w[i])) {
                v_txt.push_back(w[i]);
                is_lg = is_long_v(w[i]);
                ++i;
            }
            std::string sv = "udatta";
            bool yg = false;
            while (i < n && (w[i] == 0x0951 || w[i] == 0x0952 || is_yoga(w[i]) || w[i] == U'ऽ')) {
                if (w[i] == 0x0951) sv = "svarita";
                else if (w[i] == 0x0952) sv = "anudatta";
                else if (is_yoga(w[i])) yg = true;
                else if (w[i] == U'ऽ') is_lg = true;
                v_txt.push_back(w[i++]);
            }
            units.push_back({"CV", c_txt + v_txt, is_lg, sv, yg});
        } else if (is_indep(ch)) {
            std::u32string v_txt(1, ch);
            bool is_lg = is_long_v(ch);
            ++i;
            std::string sv = "udatta";
            bool yg = false;
            while (i < n && (w[i] == 0x0951 || w[i] == 0x0952 || is_yoga(w[i]) || w[i] == U'ऽ')) {
                if (w[i] == 0x0951) sv = "svarita";
                else if (w[i] == 0x0952) sv = "anudatta";
                else if (is_yoga(w[i])) yg = true;
                else if (w[i] == U'ऽ') is_lg = true;
                v_txt.push_back(w[i++]);
            }
            units.push_back({"V", v_txt, is_lg, sv, yg});
        } else if (ch == 0x0951 || ch == 0x0952 || ch == U'ऽ' || is_yoga(ch)) {
            if (!units.empty()) {
                units.back().text.push_back(ch);
                if (ch == 0x0951) units.back().svara = "svarita";
                else if (ch == 0x0952) units.back().svara = "anudatta";
                else if (is_yoga(ch)) units.back().yoga = true;
                else if (ch == U'ऽ') units.back().is_long = true;
            }
            ++i;
        } else {
            ++i;
        }
    }

    std::vector<AksharaInfo> aksharas;
    std::u32string onset;
    for (const auto & u : units) {
        if (u.type == "C_HAL") {
            onset += u.text;
        } else {
            int hal_cnt = 0;
            for (char32_t c : onset) if (c == 0x094D) ++hal_cnt;
            AksharaInfo ak;
            ak.ak = u32_to_utf8(onset + u.text);
            ak.is_long = u.is_long;
            ak.svara = u.svara;
            ak.has_yoga = u.yoga;
            ak.onset_hal = hal_cnt;
            ak.trailing_c = 0;
            aksharas.push_back(ak);
            onset.clear();
        }
    }
    if (!onset.empty() && !aksharas.empty()) {
        aksharas.back().ak += u32_to_utf8(onset);
        aksharas.back().trailing_c += 2;
    }
    for (size_t k = 0; k < aksharas.size(); ++k) {
        if (k + 1 < aksharas.size() && aksharas[k + 1].onset_hal >= 1) {
            aksharas[k].trailing_c += aksharas[k + 1].onset_hal + 1;
        }
        bool is_guru = aksharas[k].is_long || aksharas[k].has_yoga || aksharas[k].trailing_c >= 2 || aksharas[k].svara == "om";
        aksharas[k].weight = is_guru ? "G" : "L";
        aksharas[k].beats = is_guru ? 2.0 : 1.0;
    }
    return aksharas;
}

inline std::string detect_meter(const std::vector<AksharaInfo> & aksharas, const std::string & text) {
    size_t n = 0;
    for (const auto & a : aksharas) if (a.svara != "om") ++n;
    std::string t_norm = iast_to_deva(nfc_normalize(text));
    if (t_norm.find("भूर्भुवः") != std::string::npos || t_norm.find("तत्सवितु") != std::string::npos || (n >= 21 && n <= 26)) {
        return "Gayatri";
    }
    bool has_danda = (t_norm.find("॥") != std::string::npos || t_norm.find("।") != std::string::npos);
    bool has_ddanda = (t_norm.find("॥") != std::string::npos);
    if ((n >= 28 && n <= 36) || (n >= 14 && n <= 18 && has_danda)) return "Anushtubh";
    if ((n >= 40 && n <= 45) || (n >= 20 && n <= 23 && has_ddanda)) return "Trishtubh";
    if ((n >= 46 && n <= 52) || (n >= 24 && n <= 26 && has_ddanda)) return "Jagati";
    return "Vedic-Laya";
}

inline ChandasResult compile_chandas(const std::string & text, double bpm = 75.0, double tonic_hz = 215.0, bool add_end = true) {
    std::string t = restore_rv_svara(replace_om_re(iast_to_deva(nfc_normalize(text))));
    replace_all(t, "\xE2\x80\x8D", "");
    replace_all(t, "\xE2\x80\x8C", "");

    bool has_vedic_marks = (t.find("\xE0\xA5\x91") != std::string::npos || t.find("\xE0\xA5\x92") != std::string::npos);
    std::u32string u = utf8_to_u32(t);
    std::vector<std::u32string> raw_words;
    size_t idx = 0;
    while (idx < u.size()) {
        char32_t c = u[idx];
        if (c >= 0x0900 && c <= 0x097F && c != U'।' && c != U'॥') {
            size_t j = idx + 1;
            while (j < u.size() && u[j] >= 0x0900 && u[j] <= 0x097F && u[j] != U'।' && u[j] != U'॥') ++j;
            raw_words.push_back(u.substr(idx, j - idx));
            idx = j;
        } else {
            ++idx;
        }
    }

    std::vector<AksharaInfo> all_aksharas;
    for (const auto & rw : raw_words) {
        if (rw.find(U'ॐ') != std::u32string::npos && rw != U"ॐ") {
            std::u32string cur;
            for (char32_t c : rw) {
                if (c == U'ॐ') {
                    if (!cur.empty()) {
                        auto aks = scan_word_aksharas(u32_to_utf8(cur));
                        all_aksharas.insert(all_aksharas.end(), aks.begin(), aks.end());
                        cur.clear();
                    }
                    auto aks = scan_word_aksharas("ॐ");
                    all_aksharas.insert(all_aksharas.end(), aks.begin(), aks.end());
                } else {
                    cur.push_back(c);
                }
            }
            if (!cur.empty()) {
                auto aks = scan_word_aksharas(u32_to_utf8(cur));
                all_aksharas.insert(all_aksharas.end(), aks.begin(), aks.end());
            }
        } else {
            auto aks = scan_word_aksharas(u32_to_utf8(rw));
            all_aksharas.insert(all_aksharas.end(), aks.begin(), aks.end());
        }
    }

    std::string meter = detect_meter(all_aksharas, t);
    if (has_vedic_marks) {
        bool in_prac = false;
        for (auto & ak : all_aksharas) {
            if (ak.svara == "om") {
                ak.f0_cents = chandas_round4(203.9100 * 0.55);
                in_prac = false;
            } else if (ak.svara == "svarita") {
                ak.f0_cents = chandas_round4(203.9100);
                in_prac = true;
            } else if (ak.svara == "anudatta") {
                ak.f0_cents = chandas_round4(-203.9100);
                in_prac = false;
            } else {
                if (in_prac) {
                    ak.svara = "prachaya";
                    ak.f0_cents = chandas_round4(38.0511);
                } else {
                    ak.f0_cents = 0.0;
                }
            }
        }
    } else {
        size_t p_idx = 0;
        for (auto & ak : all_aksharas) {
            if (ak.svara == "om") {
                ak.f0_cents = chandas_round4(203.9100 * 0.65);
                continue;
            }
            size_t pos_in_pada = p_idx % 8;
            size_t pada_num = (p_idx / 8) % 4;
            bool even_pada = (pada_num % 2 == 0);
            double base_arch = 0.0;
            switch (pos_in_pada) {
                case 0: base_arch = 0.0; break;
                case 1: base_arch = 111.7313; break;
                case 2: base_arch = 203.9100; break;
                case 3: base_arch = 203.9100; break;
                case 4: base_arch = 111.7313; break;
                case 5: base_arch = even_pada ? 203.9100 : 38.0511; break;
                case 6: base_arch = even_pada ? 38.0511 : -111.7313; break;
                case 7: base_arch = even_pada ? 0.0 : -203.9100; break;
            }
            double guru_lift = (ak.weight == "G") ? 32.0 : -18.0;
            ak.f0_cents = chandas_round4(base_arch + guru_lift);
            ++p_idx;
        }
    }

    auto laya = laya_scales(t, add_end, true);
    const auto & ph = laya.ph;
    const auto & off_base = laya.off;
    size_t n_ph = ph.size();

    const double fps = 24000.0 / 256.0;
    double scale_bpm = 75.0 / std::max(bpm, 30.0);
    double laghu_ms = 160.0 * scale_bpm;

    std::vector<double> off(n_ph, 0.0);
    std::vector<double> dm(n_ph, 1.0);
    std::vector<int> dur_frames(n_ph, 5);
    std::vector<double> f0_cents_ph(n_ph, 0.0);

    size_t ak_i = 0, i = 0;
    while (i < n_ph) {
        const auto & p = ph[i];
        if (p == "o" && i + 1 < n_ph && ph[i + 1] == "m" &&
            (i == 0 || ph[i - 1] == "|" || ph[i - 1] == "," || ph[i - 1] == ";" || ph[i - 1] == ".")) {
            double c_om = (ak_i < all_aksharas.size()) ? all_aksharas[ak_i].f0_cents : 132.5415;
            if (ak_i < all_aksharas.size() && all_aksharas[ak_i].svara == "om") ++ak_i;
            off[i] = chandas_round4(c_om / 100.0 * 0.22);
            off[i + 1] = chandas_round4(c_om / 100.0 * 0.12);
            dm[i] = chandas_round4(1.60 * scale_bpm);
            dm[i + 1] = chandas_round4(1.35 * scale_bpm);
            dur_frames[i] = std::max(16, js_round(0.240 * scale_bpm * fps));
            dur_frames[i + 1] = std::max(14, js_round(0.190 * scale_bpm * fps));
            f0_cents_ph[i] = chandas_round4(c_om);
            f0_cents_ph[i + 1] = chandas_round4(c_om * 0.55);
            i += 2;
            continue;
        }
        if (starts_with_vowel_base(p)) {
            bool diph = (p == "ʌ" && i + 1 < n_ph && (ph[i + 1] == "i" || ph[i + 1] == "u"));
            bool r_voc = (i > 0 && ph[i - 1] == "r" && p == "i" && (i - 1 == 0 || !starts_with_vowel_base(ph[i - 2])));
            bool is_g = false;
            double c_val = 0.0;
            if (ak_i < all_aksharas.size()) {
                is_g = (all_aksharas[ak_i].weight == "G");
                c_val = all_aksharas[ak_i].f0_cents;
            } else {
                is_g = (p == "a" || p == "e" || p == "o" || diph);
            }
            ++ak_i;
            double sv_unit = (c_val / 100.0) * 0.62 + 0.38 * off_base[i];
            if (diph) {
                off[i] = chandas_round4(sv_unit);
                off[i + 1] = chandas_round4(sv_unit * 0.92);
                dm[i] = chandas_round4(1.08 * scale_bpm);
                dm[i + 1] = chandas_round4(1.06 * scale_bpm);
                int f_tot = std::max(12, js_round((laghu_ms * 1.85 / 1000.0) * fps));
                dur_frames[i] = f_tot / 2;
                dur_frames[i + 1] = f_tot - dur_frames[i];
                f0_cents_ph[i] = chandas_round4(c_val);
                f0_cents_ph[i + 1] = chandas_round4(c_val * 0.92);
                i += 2;
            } else {
                off[i] = chandas_round4(sv_unit);
                dm[i] = chandas_round4((is_g ? 1.14 : 0.96) * scale_bpm);
                int f_v = std::max(is_g ? 8 : 5, js_round((laghu_ms * (is_g ? 1.75 : 0.95) / 1000.0) * fps));
                if (r_voc) {
                    dur_frames[i - 1] = std::max(4, f_v / 3);
                    dur_frames[i] = std::max(5, f_v - dur_frames[i - 1]);
                } else {
                    dur_frames[i] = f_v;
                }
                f0_cents_ph[i] = chandas_round4(c_val);
                i += 1;
            }
        } else {
            if (p == "," || p == ";") {
                off[i] = chandas_round4(off_base[i]);
                dm[i] = 1.18;
                dur_frames[i] = (p == ",") ? 11 : 14;
            } else if (p == "." || p == "?" || p == "!") {
                off[i] = chandas_round4(off_base[i]);
                dm[i] = 1.20;
                dur_frames[i] = 16;
            } else if (p == "|") {
                off[i] = 0.0;
                dm[i] = 1.0;
                dur_frames[i] = 2;
            } else {
                off[i] = chandas_round4(off_base[i]);
                bool is_final_h = (p == "h" && (i + 1 == n_ph || is_punct_phone(ph[i + 1])));
                dm[i] = is_final_h ? 0.78 : 1.0;
                dur_frames[i] = is_final_h ? 5 : 6;
            }
            ++i;
        }
    }

    for (size_t k = 0; k < n_ph; ++k) {
        if (!starts_with_vowel_base(ph[k]) && !is_punct_phone(ph[k])) {
            double lv = (k > 0) ? off[k - 1] : 0.0;
            double rv = (k + 1 < n_ph) ? off[k + 1] : 0.0;
            off[k] = chandas_round4(0.5 * (lv + rv));
            double lc = (k > 0) ? f0_cents_ph[k - 1] : 0.0;
            double rc = (k + 1 < n_ph) ? f0_cents_ph[k + 1] : 0.0;
            f0_cents_ph[k] = chandas_round4(0.5 * (lc + rc));
        }
    }

    ChandasResult res;
    res.meter = meter;
    res.has_vedic_marks = has_vedic_marks;
    res.bpm = bpm;
    res.tonic_hz = tonic_hz;
    res.aksharas = std::move(all_aksharas);
    res.ph = std::move(ph);
    res.off = std::move(off);
    res.dm = std::move(dm);
    res.dur_frames = std::move(dur_frames);
    res.f0_cents = std::move(f0_cents_ph);
    return res;
}

// ---------------- SLP1 47-token Sanskrit Frontend (sakhiya/sanskrit.py phonemize_slp1) ----------------
struct Slp1Result {
    std::vector<int> ids;
    std::vector<int> weight;
    std::vector<double> off;
    std::vector<std::string> syms;
};

inline Slp1Result phonemize_slp1(const std::string & text) {
    static const std::vector<std::string> slp_symbols = {
        "<pad>", "<bos>", "<eos>", "<sp>", "<unk>", ",", "?", "!", "|", "||", "<yati>",
        "a", "A", "i", "I", "u", "U", "f", "e", "E", "o", "O",
        "k", "g", "N", "c", "j", "Y", "w", "q", "R", "t", "d", "n", "p", "b", "m", "y", "r", "l", "v", "S", "z", "s", "h", "M", "H"
    };
    static std::unordered_map<std::string, int> slp_sym2id;
    if (slp_sym2id.empty()) {
        for (size_t i = 0; i < slp_symbols.size(); ++i) slp_sym2id[slp_symbols[i]] = static_cast<int>(i);
    }
    static const std::unordered_set<std::string> slp_vowels = {"a","A","i","I","u","U","f","e","E","o","O"};
    static const std::unordered_set<std::string> slp_phones(slp_symbols.begin() + 11, slp_symbols.end());

    std::string t = nfc_normalize(text);
    // Check if [A-Za-z]{2,} in t
    bool has_two_alpha = false;
    for (size_t i = 0; i + 1 < t.size(); ++i) {
        if (is_ascii_alpha(static_cast<unsigned char>(t[i])) && is_ascii_alpha(static_cast<unsigned char>(t[i + 1]))) {
            has_two_alpha = true;
            break;
        }
    }
    if (has_two_alpha) t = iast_to_deva(t);
    replace_all(t, "।।", "॥");
    replace_all(t, "||", "॥");

    // Strip parenthesized (...)
    std::string t2;
    int paren = 0;
    for (char c : t) {
        if (c == '(') { ++paren; t2.push_back(' '); }
        else if (c == ')') { if (paren > 0) --paren; }
        else if (paren == 0) t2.push_back(c);
    }
    t = t2;

    // Strip verse numbers after danda: ([।॥])\s*[0-9०-९]+(?:\s*[/.\-]\s*[0-9०-९]+)*\s*(?=[।॥]|$)
    std::u32string u = utf8_to_u32(t);
    auto is_num_cp = [](char32_t c) { return (c >= U'0' && c <= U'9') || (c >= 0x0966 && c <= 0x096F); };
    std::u32string u_clean;
    for (size_t i = 0; i < u.size();) {
        u_clean.push_back(u[i]);
        if (u[i] == U'।' || u[i] == U'॥') {
            size_t j = i + 1;
            while (j < u.size() && (u[j] == U' ' || u[j] == U'\t')) ++j;
            if (j < u.size() && is_num_cp(u[j])) {
                while (j < u.size() && (is_num_cp(u[j]) || u[j] == U' ' || u[j] == U'\t' || u[j] == U'/' || u[j] == U'.' || u[j] == U'-')) ++j;
                if (j == u.size() || u[j] == U'।' || u[j] == U'॥') {
                    i = j;
                    continue;
                }
            }
        }
        ++i;
    }
    t = u32_to_utf8(u_clean);
    replace_all(t, " - ", " , ");
    replace_all(t, " -- ", " , ");
    t.erase(std::remove(t.begin(), t.end(), '-'), t.end());
    replace_all(t, ":", " , ");
    t = trim_str(t);

    bool has_acc = (t.find("\xE0\xA5\x91") != std::string::npos || t.find("\xE0\xA5\x92") != std::string::npos);

    static const std::unordered_map<char32_t, std::string> slp_ind = {
        {U'अ',"a"},{U'आ',"A"},{U'इ',"i"},{U'ई',"I"},{U'उ',"u"},{U'ऊ',"U"},{U'ऋ',"f"},{U'ॠ',"F"},
        {U'ऌ',"x"},{U'ॡ',"X"},{U'ए',"e"},{U'ऐ',"E"},{U'ओ',"o"},{U'औ',"O"}
    };
    static const std::unordered_map<char32_t, std::string> slp_sgn = {
        {U'ा',"A"},{U'ि',"i"},{U'ी',"I"},{U'ु',"u"},{U'ू',"U"},{U'ृ',"f"},{U'ॄ',"F"},
        {U'ॢ',"x"},{U'ॣ',"X"},{U'े',"e"},{U'ै',"E"},{U'ो',"o"},{U'ौ',"O"}
    };
    static const std::unordered_map<char32_t, std::string> slp_con = {
        {U'क',"k"},{U'ख',"K"},{U'ग',"g"},{U'घ',"G"},{U'ङ',"N"},{U'च',"c"},{U'छ',"C"},{U'ज',"j"},{U'झ',"J"},{U'ञ',"Y"},
        {U'ट',"w"},{U'ठ',"W"},{U'ड',"q"},{U'ढ',"Q"},{U'ण',"R"},{U'त',"t"},{U'थ',"T"},{U'द',"d"},{U'ध',"D"},{U'न',"n"},
        {U'प',"p"},{U'फ',"P"},{U'ब',"b"},{U'भ',"B"},{U'म',"m"},{U'य',"y"},{U'र',"r"},{U'ल',"l"},{U'व',"v"},{U'श',"S"},
        {U'ष',"z"},{U'स',"s"},{U'ह',"h"},{U'ळ',"l"}
    };
    static const std::unordered_map<char32_t, std::string> slp_marks = {
        {U'।',"|"},{U'|',"|"},{U'.',"|"},{U'॥',"||"},{U',',","},{U';',","},{U'…',","},{U'–',","},{U'—',","},{U'?',"?"},{U'!',"!"}
    };
    static const std::unordered_map<std::string, std::string> slp_nasal = {
        {"k","N"},{"K","N"},{"g","N"},{"G","N"},{"N","N"},
        {"c","Y"},{"C","Y"},{"j","Y"},{"J","Y"},{"Y","Y"},
        {"w","R"},{"W","R"},{"q","R"},{"Q","R"},{"R","R"},
        {"t","n"},{"T","n"},{"d","n"},{"D","n"},{"n","n"},
        {"p","m"},{"P","m"},{"b","m"},{"B","m"},{"m","m"}
    };
    static const std::unordered_map<std::string, std::vector<std::string>> slp_written = {
        {"F",{"f"}},{"x",{"l","f"}},{"X",{"l","f"}},
        {"K",{"k","h"}},{"G",{"g","h"}},{"C",{"c","h"}},{"J",{"j","h"}},
        {"W",{"w","h"}},{"Q",{"q","h"}},{"T",{"t","h"}},{"D",{"d","h"}},{"P",{"p","h"}},{"B",{"b","h"}}
    };
    static const std::unordered_set<std::string> slp_raw_v = {"a","A","i","I","u","U","f","F","x","X","e","E","o","O"};
    static const std::unordered_set<std::string> slp_long = {"A","I","U","F","X","e","E","o","O"};

    struct Piece {
        bool is_mark = false;
        std::string mark;
        std::vector<std::string> phs;
        std::vector<double> accs;
    };

    // Split by whitespace or _SLP_MARKS
    u = utf8_to_u32(t);
    std::vector<std::u32string> raw_pcs;
    size_t p_i = 0;
    while (p_i < u.size()) {
        char32_t c = u[p_i];
        if (c == U' ' || c == U'\t' || c == U'\n' || c == U'\r') {
            ++p_i;
        } else if (slp_marks.count(c)) {
            raw_pcs.push_back(std::u32string(1, c));
            ++p_i;
        } else {
            size_t j = p_i + 1;
            while (j < u.size() && u[j] != U' ' && u[j] != U'\t' && u[j] != U'\n' && u[j] != U'\r' && !slp_marks.count(u[j])) ++j;
            raw_pcs.push_back(u.substr(p_i, j - p_i));
            p_i = j;
        }
    }

    auto is_slp_read = [&](char32_t c) {
        return slp_ind.count(c) || slp_sgn.count(c) || slp_con.count(c) ||
               c == U'ं' || c == U'ँ' || c == U'ः' || c == 0x1CF5 || c == 0x1CF6 ||
               c == U'्' || c == U'़' || c == U'ऽ' || c == U'ॐ' || c == U' ';
    };

    std::vector<Piece> pieces;
    for (const auto & pc : raw_pcs) {
        if (pc.size() == 1 && slp_marks.count(pc[0])) {
            Piece pm; pm.is_mark = true; pm.mark = slp_marks.at(pc[0]);
            pieces.push_back(pm);
            continue;
        }
        std::vector<std::string> phs;
        std::vector<double> accs;
        bool open_c = false, stray = false;
        std::u32string pc_sp = pc + U" ";
        for (char32_t ch : pc_sp) {
            if (ch == 0x0951 || ch == 0x0952) {
                double v = (ch == 0x0951) ? 1.0 : -1.0;
                if (open_c) {
                    phs.push_back("a"); accs.push_back(v); open_c = false;
                } else if (!accs.empty()) {
                    accs.back() = v;
                }
                continue;
            }
            if (ch == 0x200B || ch == 0x200C || ch == 0x200D || ch == 0x0953 || ch == 0x0954) continue;
            if (!is_slp_read(ch)) {
                if (open_c) { phs.push_back("a"); accs.push_back(0.0); open_c = false; }
                if (!stray) { phs.push_back("<unk>"); accs.push_back(0.0); stray = true; }
                continue;
            }
            stray = false;
            if (ch == U'्') {
                open_c = false;
            } else if (ch == U'़') {
                // pass
            } else if (slp_sgn.count(ch)) {
                phs.push_back(slp_sgn.at(ch)); accs.push_back(0.0); open_c = false;
            } else {
                if (open_c) { phs.push_back("a"); accs.push_back(0.0); }
                open_c = (slp_con.count(ch) > 0);
                if (open_c) {
                    phs.push_back(slp_con.at(ch)); accs.push_back(0.0);
                } else if (slp_ind.count(ch)) {
                    phs.push_back(slp_ind.at(ch)); accs.push_back(0.0);
                } else if (ch == U'ं' || ch == U'ँ') {
                    phs.push_back("M"); accs.push_back(0.0);
                } else if (ch == U'ः' || ch == 0x1CF5 || ch == 0x1CF6) {
                    phs.push_back("H"); accs.push_back(0.0);
                } else if (ch == U'ऽ') {
                    for (int k_i = static_cast<int>(phs.size()) - 1; k_i >= 0; --k_i) {
                        if (slp_raw_v.count(phs[k_i])) {
                            if (phs[k_i] == "a") phs[k_i] = "A";
                            else if (phs[k_i] == "i") phs[k_i] = "I";
                            else if (phs[k_i] == "u") phs[k_i] = "U";
                            else if (phs[k_i] == "f") phs[k_i] = "F";
                            else if (phs[k_i] == "x") phs[k_i] = "X";
                            break;
                        }
                    }
                } else if (ch == U'ॐ') {
                    phs.push_back("o"); phs.push_back("m");
                    accs.push_back(0.0); accs.push_back(0.0);
                }
            }
        }
        if (!phs.empty()) {
            Piece pw; pw.is_mark = false; pw.phs = std::move(phs); pw.accs = std::move(accs);
            pieces.push_back(pw);
        }
    }

    for (size_t idx = 0; idx < pieces.size(); ++idx) {
        if (pieces[idx].is_mark) continue;
        auto & phs = pieces[idx].phs;
        for (size_t i = 0; i < phs.size(); ++i) {
            if (phs[i] == "M") {
                if (i + 1 < phs.size()) {
                    auto it = slp_nasal.find(phs[i + 1]);
                    phs[i] = (it != slp_nasal.end()) ? it->second : "M";
                } else if (idx + 1 == pieces.size() || pieces[idx + 1].is_mark) {
                    phs[i] = "m";
                }
            }
        }
    }

    std::vector<std::string> syms = {"<bos>"};
    std::vector<int> wt = {0};
    std::vector<double> off = {0.0};
    size_t st = 0;
    while (st < pieces.size()) {
        if (pieces[st].is_mark) {
            if (syms.back() != pieces[st].mark) {
                syms.push_back(pieces[st].mark); wt.push_back(0); off.push_back(0.0);
            }
            ++st;
            continue;
        }
        size_t en = st;
        while (en < pieces.size() && !pieces[en].is_mark) ++en;
        std::vector<std::string> flat_p;
        for (size_t k = st; k < en; ++k) {
            for (const auto & p : pieces[k].phs) flat_p.push_back(p);
        }
        std::vector<int> w_list(flat_p.size(), 0);
        std::vector<size_t> v_idx;
        for (size_t i = 0; i < flat_p.size(); ++i) {
            if (slp_raw_v.count(flat_p[i])) v_idx.push_back(i);
        }
        for (size_t n_i = 0; n_i < v_idx.size(); ++n_i) {
            size_t vi = v_idx[n_i];
            size_t end_btw = (n_i + 1 < v_idx.size()) ? v_idx[n_i + 1] : flat_p.size();
            size_t btw_len = end_btw - (vi + 1);
            bool has_mh = false;
            for (size_t b = vi + 1; b < end_btw; ++b) {
                if (flat_p[b] == "M" || flat_p[b] == "H") has_mh = true;
            }
            w_list[vi] = (slp_long.count(flat_p[vi]) || btw_len >= 2 || has_mh) ? 2 : 1;
        }
        int yati_k = -1;
        size_t n_seg = en - st;
        if (t.find("॥") != std::string::npos && v_idx.size() >= 14 && v_idx.size() <= 18 && n_seg > 1) {
            std::vector<int> cum(n_seg, 0);
            int acc_sy = 0;
            for (size_t k = 0; k < n_seg; ++k) {
                for (const auto & p : pieces[st + k].phs) if (slp_raw_v.count(p)) ++acc_sy;
                cum[k] = acc_sy;
            }
            size_t bk = 0;
            for (size_t k = 1; k + 1 < n_seg; ++k) {
                if (std::abs(cum[k] - 8) < std::abs(cum[bk] - 8)) bk = k;
            }
            if (std::abs(cum[bk] - 8) <= 2) yati_k = static_cast<int>(bk);
        }
        size_t w_iter = 0;
        for (size_t wk = 0; wk < n_seg; ++wk) {
            if (slp_phones.count(syms.back()) || syms.back() == "<unk>") {
                syms.push_back("<sp>"); wt.push_back(0); off.push_back(0.0);
            }
            int prac = 0;
            const auto & w_p = pieces[st + wk].phs;
            const auto & w_a = pieces[st + wk].accs;
            for (size_t i = 0; i < w_p.size(); ++i) {
                int lv = w_list[w_iter++];
                std::vector<std::string> items;
                auto itW = slp_written.find(w_p[i]);
                if (itW != slp_written.end()) items = itW->second;
                else items = {w_p[i]};
                for (const auto & item : items) {
                    syms.push_back(item);
                    bool is_v = (slp_vowels.count(item) > 0);
                    wt.push_back(is_v ? lv : 0);
                    if (is_v) {
                        if (w_a[i] == 1.0) { off.push_back(1.08); prac = 1; }
                        else if (w_a[i] == -1.0) { off.push_back(-1.15); prac = 0; }
                        else if (prac == 1) { off.push_back(0.38); }
                        else { off.push_back(0.0); }
                    } else {
                        off.push_back(0.0);
                    }
                }
            }
            if (static_cast<int>(wk) == yati_k) {
                syms.push_back("<yati>"); wt.push_back(0); off.push_back(0.0);
            }
        }
        st = en;
    }
    syms.push_back("<eos>"); wt.push_back(0); off.push_back(0.0);

    bool has_yati = false;
    for (const auto & s : syms) if (s == "<yati>") has_yati = true;
    if (has_acc || has_yati) {
        for (size_t i = 0; i < syms.size(); ++i) {
            const auto & s = syms[i];
            double c_v = 0.0;
            bool apply = false;
            if (has_acc && (s == "," || s == "|" || s == "||" || s == "?" || s == "!")) {
                c_v = (s == ",") ? 0.5 : ((s == "|" || s == "||") ? -0.7 : (s == "?" ? 0.8 : -0.2));
                apply = true;
            } else if (!has_acc && (s == "<yati>" || s == "|" || s == "||")) {
                c_v = (s == "<yati>") ? 0.35 : (s == "|" ? 0.20 : -0.40);
                apply = true;
            }
            if (!apply) continue;
            std::vector<int> js;
            int min_j = std::max(-1, static_cast<int>(i) - 7);
            for (int j = static_cast<int>(i) - 1; j > min_j && js.size() < 2; --j) {
                if (slp_vowels.count(syms[j])) js.push_back(j);
            }
            for (size_t n_j = 0; n_j < js.size(); ++n_j) {
                off[js[n_j]] += c_v * (n_j == 0 ? 1.0 : 0.4);
            }
        }
        std::vector<double> sm = off;
        for (size_t i = 0; i < syms.size(); ++i) {
            if (slp_phones.count(syms[i]) && !slp_vowels.count(syms[i])) {
                double lv = (i > 0) ? off[i - 1] : 0.0;
                double rv = (i + 1 < off.size()) ? off[i + 1] : 0.0;
                sm[i] = 0.5 * (lv + rv);
            }
        }
        off = std::move(sm);
    }

    Slp1Result res;
    res.ids.reserve(syms.size());
    for (const auto & s : syms) {
        auto it = slp_sym2id.find(s);
        res.ids.push_back(it != slp_sym2id.end() ? it->second : 4);
    }
    res.weight = std::move(wt);
    res.off = std::move(off);
    res.syms = std::move(syms);
    return res;
}

// ---------------- Sentence Segmentation & Mixed Language Routing ----------------
inline std::vector<std::string> segment_text(const std::string & text) {
    static const std::unordered_set<std::string> finite = {
        "छ", "छन्", "छु", "छौ", "छैन", "छैनन्", "हो", "हुन्", "थियो", "थिए", "थिएन", "थिएनन्", "भयो", "भए", "गर्यो", "गरे", "गर्छ", "गर्छन्",
        "हुन्छ", "हुन्छन्", "पर्छ", "पर्दछ", "सक्छ", "सकिन्छ", "गयो", "आयो", "दियो", "लियो", "भन्यो", "बताए", "पाइन्छ", "मानिन्छ", "गरियो", "देखिन्छ", "रहेछ",
        "बा", "बाटै", "बाटु", "रहे", "है", "नाहै", "बडयी", "बडिय", "हलयी"
    };

    // Split by (?<=[।॥])\s*|(?<=[.?!])\s+
    std::u32string u = utf8_to_u32(trim_str(text));
    std::vector<std::string> sents;
    std::u32string cur_s;
    size_t i = 0;
    while (i < u.size()) {
        char32_t c = u[i];
        cur_s.push_back(c);
        if (c == U'।' || c == U'॥') {
            std::string st = trim_str(u32_to_utf8(cur_s));
            if (!st.empty()) sents.push_back(st);
            cur_s.clear();
            while (i + 1 < u.size() && (u[i + 1] == U' ' || u[i + 1] == U'\t' || u[i + 1] == U'\n' || u[i + 1] == U'\r')) ++i;
        } else if ((c == U'.' || c == U'?' || c == U'!') && i + 1 < u.size() &&
                   (u[i + 1] == U' ' || u[i + 1] == U'\t' || u[i + 1] == U'\n' || u[i + 1] == U'\r')) {
            std::string st = trim_str(u32_to_utf8(cur_s));
            if (!st.empty()) sents.push_back(st);
            cur_s.clear();
            while (i + 1 < u.size() && (u[i + 1] == U' ' || u[i + 1] == U'\t' || u[i + 1] == U'\n' || u[i + 1] == U'\r')) ++i;
        }
        ++i;
    }
    std::string rem = trim_str(u32_to_utf8(cur_s));
    if (!rem.empty()) sents.push_back(rem);

    std::vector<std::string> out;
    for (const auto & s : sents) {
        std::istringstream iss(s);
        std::vector<std::string> ws;
        std::string w;
        while (iss >> w) ws.push_back(w);
        if (ws.size() <= 20) {
            out.push_back(s);
            continue;
        }
        std::vector<std::string> cur;
        for (size_t k = 0; k < ws.size(); ++k) {
            cur.push_back(ws[k]);
            if ((finite.count(ws[k]) && cur.size() >= 4 && (ws.size() - k - 1) >= 3) || cur.size() >= 12) {
                std::string joined;
                for (size_t m = 0; m < cur.size(); ++m) joined += (m ? " " : "") + cur[m];
                out.push_back(joined);
                cur.clear();
            }
        }
        if (!cur.empty()) {
            std::string joined;
            for (size_t m = 0; m < cur.size(); ++m) joined += (m ? " " : "") + cur[m];
            if (!out.empty() && cur.size() < 3) out.back() += " " + joined;
            else out.push_back(joined);
        }
    }
    if (out.empty()) out.push_back(text);
    return out;
}

inline std::vector<std::pair<std::string, std::string>> route_mixed(const std::string & text, const std::string & tharu = "thr_dangaura") {
    static const std::unordered_set<std::string> ne_visarga = {
        "अतः", "प्रायः", "पुनः", "मूलतः", "प्रातः", "सम्भवतः", "संभवतः", "विशेषतः", "वस्तुतः", "अन्ततः", "क्रमशः",
        "मुख्यतः", "प्रायशः", "स्वतः", "सर्वप्रथमतः", "प्रथमतः", "अन्तःकरण"
    };
    static const std::unordered_set<std::string> tharu_strong = {
        "संघरियाहुक्रे", "संघरिया", "संघरियाहुक्रेन", "बनाइल", "बाटै", "बाटु", "बाटी", "बडयी", "बडिय", "हलयी",
        "नाहै", "तोहार", "तोहर", "मोर", "हमाल", "हमार", "हुक्रे", "कहथ", "जाथ", "खाइल", "गइल", "रहल"
    };
    static const std::unordered_set<std::string> tharu_weak = {"हों", "बा", "मै", "जी", "छोट", "हइ"};
    static const std::u32string sa_hard = U"ॐ॥\u0951\u0952ऽāīūṛṝḷṅñṭḍṇśṣṃṁḥ";

    std::u32string u = nfc_normalize_u32(utf8_to_u32(text));
    std::vector<std::string> clauses;
    std::u32string cur;
    for (size_t i = 0; i < u.size(); ++i) {
        char32_t c = u[i];
        if (c == U'\n') {
            std::string st = trim_str(u32_to_utf8(cur));
            if (!st.empty()) clauses.push_back(st);
            cur.clear();
        } else {
            cur.push_back(c);
            if (c == U'।' || c == U'॥' || c == U'!' || c == U'?' ||
                (c == U'.' && (i + 1 == u.size() || u[i + 1] == U' ' || u[i + 1] == U'\t' || u[i + 1] == U'\n'))) {
                std::string st = trim_str(u32_to_utf8(cur));
                if (!st.empty()) clauses.push_back(st);
                cur.clear();
            }
        }
    }
    std::string rem = trim_str(u32_to_utf8(cur));
    if (!rem.empty()) clauses.push_back(rem);

    auto detect_lang = [&](const std::string & clause) -> std::string {
        std::u32string cu = nfc_normalize_u32(utf8_to_u32(clause));
        for (char32_t c : cu) {
            if (sa_hard.find(c) != std::u32string::npos) return "sa";
        }
        std::vector<std::string> words;
        size_t idx = 0;
        while (idx < cu.size()) {
            char32_t c = cu[idx];
            if ((c >= 0x0900 && c <= 0x0963) || (c >= 0x0966 && c <= 0x097F)) {
                size_t j = idx + 1;
                while (j < cu.size() && ((cu[j] >= 0x0900 && cu[j] <= 0x0963) || (cu[j] >= 0x0966 && cu[j] <= 0x097F))) ++j;
                words.push_back(u32_to_utf8(cu.substr(idx, j - idx)));
                idx = j;
            } else {
                ++idx;
            }
        }
        int strong = 0, weak = 0, vis = 0, halm = 0;
        for (const auto & w : words) {
            if (tharu_strong.count(w)) ++strong;
            if (tharu_weak.count(w)) ++weak;
            if (str_ends_with(w, "ः") && !ne_visarga.count(w)) ++vis;
            if (str_ends_with(w, "म्") && utf8_to_u32(w).size() > 3) ++halm;
        }
        if (strong >= 1 || weak >= 3) return tharu;
        if (vis > 0 || halm >= 2) return "sa";
        return "ne";
    };

    std::vector<std::pair<std::string, std::string>> segs;
    for (const auto & c : clauses) {
        std::string L = detect_lang(c);
        if (!segs.empty() && segs.back().first == L) {
            segs.back().second += " " + c;
        } else {
            segs.push_back({L, c});
        }
    }
    return segs;
}

struct StyleConfig {
    int style_id;
    float rate;
    float pitch_shift;
    float pitch_scale;
    float energy_scale;
    float pause_scale;
};

inline StyleConfig get_style_config(const std::string & style) {
    static const std::unordered_map<std::string, StyleConfig> styles = {
        {"neutral",        {0, 0.96f,  0.00f, 1.12f, 1.00f, 1.00f}},
        {"news",           {0, 0.96f, -0.03f, 1.12f, 1.06f, 1.00f}},
        {"conversational", {1, 1.00f,  0.04f, 1.32f, 1.02f, 0.88f}},
        {"storytelling",   {2, 0.90f,  0.02f, 1.32f, 1.04f, 1.35f}},
        {"warm_soft",      {3, 0.96f, -0.03f, 1.18f, 0.92f, 1.15f}},
        {"slow_clear",     {3, 0.78f,  0.00f, 1.00f, 1.00f, 1.30f}},
        {"expressive",     {4, 1.00f,  0.08f, 1.32f, 1.08f, 0.92f}},
        {"vedic_chant",    {5, 0.90f,  0.00f, 0.92f, 1.04f, 1.28f}},
        {"chant",          {5, 0.90f,  0.00f, 0.92f, 1.04f, 1.28f}}
    };
    auto it = styles.find(style);
    return (it != styles.end()) ? it->second : styles.at("neutral");
}

} // namespace sakhiya
