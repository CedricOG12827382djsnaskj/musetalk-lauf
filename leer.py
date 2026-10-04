# Entfernt die markierte Fläche aus einem Bild und füllt sie passend zur Umgebung (LaMa, Apache 2.0).
# Eingabe: auftrag/bild.png, auftrag/maske.png (weiß = füllen). Ausgabe: ergebnis/leer.png in Originalgröße.
import os, sys, time
from PIL import Image
from simple_lama_inpainting import SimpleLama

A, E = sys.argv[1], sys.argv[2]
os.makedirs(E, exist_ok=True)
t0 = time.time()
lama = SimpleLama()
bild = Image.open(f'{A}/bild.png').convert('RGB')
maske = Image.open(f'{A}/maske.png').convert('L')
erg = lama(bild, maske).crop((0, 0, bild.width, bild.height))
erg.save(f'{E}/leer.png')
print(f'{time.time() - t0:.0f} s  leer.png {erg.width}x{erg.height}', flush=True)
