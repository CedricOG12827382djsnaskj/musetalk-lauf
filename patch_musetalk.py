# Patch fuer MuseTalk scripts/inference.py: Bilder ohne erkanntes Gesicht (etwa wenn sich der Sprecher umdreht)
# nicht ueberspringen, sonst reisst die Bildfolge ab und das Video wird abgeschnitten. Sie bleiben unveraendert.
p = 'scripts/inference.py'
s = open(p, encoding='utf-8').read()

a = '            print(f"Number of frames: {len(frame_list)}")'
assert a in s, 'Ankerzeile 1 fehlt'
s = s.replace(a, '''            ohne_gesicht = set(k for k, b in enumerate(coord_list) if b == coord_placeholder)
            gueltig = [b for b in coord_list if b != coord_placeholder]
            if gueltig:
                letzte = gueltig[0]
                neu = []
                for b in coord_list:
                    if b == coord_placeholder:
                        neu.append(letzte)
                    else:
                        letzte = b
                        neu.append(b)
                coord_list = neu
            print(f"Bilder ohne Gesicht: {len(ohne_gesicht)}")
''' + a, 1)

b = '                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)'
assert b in s, 'Ankerzeile 2 fehlt'
s = s.replace(b, '''                n_bilder = len(frame_list)
                k = i % (2 * n_bilder)
                k = k if k < n_bilder else 2 * n_bilder - 1 - k
                if k in ohne_gesicht:
                    combine_frame = ori_frame
''' + b, 1)

c = '''                except:
                    continue'''
assert c in s, 'Ankerzeile 3 fehlt'
s = s.replace(c, '''                except:
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                    continue''', 1)
open(p, 'w', encoding='utf-8').write(s)
print('MuseTalk gepatcht')
