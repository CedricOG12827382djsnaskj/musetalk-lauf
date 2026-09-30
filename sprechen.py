# Spricht Sätze mit einer geklonten Stimme (Qwen3-TTS Base) und liefert Wortzeiten (Qwen3-ForcedAligner).
# Eingabe: auftrag/ref.wav (oder auftrag/entwurf.txt mit einer Stimmbeschreibung), auftrag/ref.txt, auftrag/saetze.json [{"id": "...", "text": "..."}]
# Ausgabe: ergebnis/<id>.wav und ergebnis/<id>.woerter.json [{"wort", "von", "bis"}]
import json, os, sys, time
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
from qwen_asr import Qwen3ForcedAligner

torch.set_num_threads(os.cpu_count())
A, E = sys.argv[1], sys.argv[2]
os.makedirs(E, exist_ok=True)
modell = os.environ.get('TTS_MODELL', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base')
saetze = json.load(open(f'{A}/saetze.json', encoding='utf-8'))
ref_text = open(f'{A}/ref.txt', encoding='utf-8').read().strip()

t0 = time.time()
# Optional: Stimme erst entwerfen (Qwen3-TTS VoiceDesign), wenn statt ref.wav eine Beschreibung in entwurf.txt liegt.
# Der Entwurf spricht ref.txt und wird danach wie eine normale Referenz geklont; er liegt auch in ergebnis/ref.wav.
if os.path.exists(f'{A}/entwurf.txt') and not os.path.exists(f'{A}/ref.wav'):
    beschreibung = open(f'{A}/entwurf.txt', encoding='utf-8').read().strip()
    vd = Qwen3TTSModel.from_pretrained(os.environ.get('TTS_ENTWURF', 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'), device_map='cpu', dtype=torch.float32)
    wavs, sr = vd.generate_voice_design(text=ref_text, language='German', instruct=beschreibung)
    sf.write(f'{A}/ref.wav', wavs[0], sr)
    sf.write(f'{E}/ref.wav', wavs[0], sr)
    print(f"{time.time() - t0:6.0f} s  Stimme entworfen", flush=True)
    del vd
tts = Qwen3TTSModel.from_pretrained(modell, device_map='cpu', dtype=torch.float32)
prompt = tts.create_voice_clone_prompt(ref_audio=f'{A}/ref.wav', ref_text=ref_text)
for s in saetze:
    wavs, sr = tts.generate_voice_clone(text=s['text'], language='German', voice_clone_prompt=prompt)
    sf.write(f"{E}/{s['id']}.wav", wavs[0], sr)
    print(f"{time.time() - t0:6.0f} s  gesprochen {s['id']}", flush=True)
del tts

al = Qwen3ForcedAligner.from_pretrained('Qwen/Qwen3-ForcedAligner-0.6B', device_map='cpu', dtype=torch.float32)
for s in saetze:
    r = al.align(audio=f"{E}/{s['id']}.wav", text=s['text'], language='German')
    woerter = [{'wort': w.text, 'von': float(w.start_time), 'bis': float(w.end_time)} for w in r[0]]
    json.dump(woerter, open(f"{E}/{s['id']}.woerter.json", 'w', encoding='utf-8'), ensure_ascii=False)
print(f"{time.time() - t0:6.0f} s  fertig", flush=True)
