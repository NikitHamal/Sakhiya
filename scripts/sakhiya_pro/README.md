# Sakhiya-Pro (large Nepal-native teacher TTS)
Box setup + data + bake-off + fine-tune + render scripts for the 98 GB Blackwell box.
Tokens are read from /tmp/hf_token and /tmp/gitenv.sh on the box only; never committed.
Note: ai4bharat/IndicF5 is gated and the project HF token is not authorised for it (403), so the
bake-off uses Swar (an IndicF5 fine-tune with identical F5-Base architecture) as the F5 representative.
