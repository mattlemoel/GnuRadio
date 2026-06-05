#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  kiwi_rt.py
#  Decodeur TEMPS REEL de telemesure Kiwi / Kikiwi, directement sur RTL-SDR
# -----------------------------------------------------------------------------
#  Auteur      : Matthieu Le Moel
#  Annee       : 2026
#
#  LICENCES (double licence)
#  -------------------------
#  CODE SOURCE : GNU General Public License v3 ou ulterieure (GPL-3.0-or-later)
#    Copyright (C) 2026  Matthieu Le Moel
#    Logiciel libre, SANS AUCUNE GARANTIE. Voir <https://www.gnu.org/licenses/>.
#  DOCUMENTATION / COMMENTAIRES : Creative Commons CC BY-SA 4.0
#    <https://creativecommons.org/licenses/by-sa/4.0/deed.fr>
# =============================================================================
#
#  ROLE
#  ----
#  Contrairement a kiwi_fm_rx.py (qui enregistre un WAV via GNU Radio) et
#  kiwi_decoder.py (qui decode un WAV), ce programme fait TOUT en direct et en
#  Python pur : il lit les echantillons IQ du RTL-SDR, demodule la FM lui-meme,
#  puis decode l'AFSK et affiche les trames au fil de l'eau.
#
#  CHAINE TEMPS REEL :
#
#     RTL-SDR (IQ)  ->  filtre canal + decimation  ->  demod FM (phase)
#                   ->  audio 48 kHz  ->  AFSK -> UART -> trames  ->  affichage
#
#  La partie "AFSK -> trames" reutilise telles quelles les fonctions DEJA
#  VALIDEES de kiwi_decoder.py (importe comme module). Seule la partie radio
#  (lecture IQ + demod FM) est nouvelle ici.
#
#  PREREQUIS :
#     pip install pyrtlsdr numpy scipy
#     (et la bibliotheque systeme librtlsdr : sudo apt install rtl-sdr librtlsdr-dev)
#
#  UTILISATION :
#     python3 kiwi_rt.py            # decodage temps reel depuis le dongle
#     python3 kiwi_rt.py --test     # autotest sans materiel (IQ synthetique)
# =============================================================================

import sys
import numpy as np
from scipy.signal import firwin, lfilter

# Reutilisation du decodeur deja valide (memes constantes : BAUD, MARK_HZ...).
import kiwi_decoder as kd


# =============================================================================
#  PARAMETRES RADIO AJUSTABLES
# =============================================================================
CENTER_FREQ = 137.950e6   # frequence de l'emetteur (ou 138.500e6)
FS_IN       = 240_000     # debit IQ du RTL-SDR (Hz) ; 240k = 5 x 48k (decim entiere)
FS_AUDIO    = 48_000      # debit audio en sortie de demod (= entree du decodeur)
CHAN_BW     = 12_000      # largeur du filtre canal avant decimation (Hz)
RF_GAIN     = 'auto'      # gain tuner : 'auto' ou un nombre en dB (ex : 40)
PPM         = 0           # correction de derive du dongle (parties par million)
CHUNK       = 262_144     # nb d'echantillons IQ lus par iteration (~1.1 s @240k)


# =============================================================================
#  RECEPTEUR FM EN STREAMING
#  -------------------------
#  Conserve l'etat entre les blocs pour assurer la continuite :
#    - zi  : etat du filtre canal (pas de discontinuite en bord de bloc)
#    - last: dernier echantillon IQ (continuite de la demod par difference de phase)
# =============================================================================
class FMReceiver:
    def __init__(self, fs_in, fs_audio, chan_bw):
        self.fs_in = fs_in
        self.fs_audio = fs_audio
        self.decim = int(round(fs_in / fs_audio))   # facteur de decimation entier

        # Filtre passe-bas anti-repliement (FIR) avant decimation.
        self.taps = firwin(numtaps=129, cutoff=chan_bw, fs=fs_in)
        # Etat initial du filtre (complexe car le signal IQ est complexe).
        self.zi = np.zeros(len(self.taps) - 1, dtype=complex)
        # Dernier echantillon IQ du bloc precedent (pour la difference de phase).
        self.last = 0.0 + 0.0j

    def process(self, iq):
        """IQ brut (complexe) -> audio demodule (reel) a fs_audio."""
        # 1) Filtrage canal (garde uniquement la bande NBFM), avec etat continu.
        filt, self.zi = lfilter(self.taps, 1.0, iq, zi=self.zi)

        # 2) Decimation : on ne garde qu'un echantillon sur 'decim'.
        x = filt[::self.decim]
        if len(x) == 0:
            return np.array([], dtype=float)

        # 3) Demodulation FM : la frequence instantanee = phase du produit de
        #    l'echantillon courant par le conjugue du precedent. On prepend le
        #    dernier echantillon du bloc precedent pour la continuite.
        prev = np.empty(len(x), dtype=complex)
        prev[0] = self.last
        prev[1:] = x[:-1]
        self.last = x[-1]
        audio = np.angle(x * np.conj(prev))   # signal audio (tonalites AFSK)
        return audio


# =============================================================================
#  DECODAGE D'UN BLOC AUDIO -> TRAMES (via le module deja valide)
# =============================================================================
def decode_audio(audio, fs_audio):
    """audio -> trames Kiwi, en reutilisant les fonctions de kiwi_decoder."""
    if len(audio) < fs_audio // 10:        # bloc trop court : on ignore
        return []
    metric = kd.afsk_metric(audio, fs_audio)
    by = kd.metric_to_bytes(metric, fs_audio)
    return kd.parse_frames(by)


def afficher_trames(frames, vues):
    """Affiche les trames valides, en regroupant les repetitions identiques.

    'vues' est un dict d'etat conserve entre appels {signature: compte}.
    """
    for f in frames:
        if not f["valide"]:
            continue
        sig = (tuple(f["voies_raw"]), f["alim_raw"])
        if sig == vues.get("derniere"):
            vues["compte"] += 1
            # reaffiche la meme ligne en mettant a jour le compteur de repetitions
            print(f"\r  (x{vues['compte']}) voies={f['voies_volts']} "
                  f"Vbat~{f['vbat_volts']}V", end="", flush=True)
        else:
            vues["derniere"] = sig
            vues["compte"] = 1
            print(f"\n[OK] voies={f['voies_volts']} V  |  Vbat~{f['vbat_volts']} V",
                  end="", flush=True)


# =============================================================================
#  MODE TEMPS REEL : LECTURE DIRECTE DU RTL-SDR
# =============================================================================
def temps_reel():
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        sys.exit("pyrtlsdr manquant : pip install pyrtlsdr "
                 "(+ sudo apt install rtl-sdr librtlsdr-dev)")

    sdr = RtlSdr()
    sdr.sample_rate = FS_IN
    sdr.center_freq = CENTER_FREQ
    sdr.freq_correction = PPM if PPM != 0 else 1   # ppm (1 minimum si 0)
    sdr.gain = RF_GAIN

    rx = FMReceiver(FS_IN, FS_AUDIO, CHAN_BW)
    vues = {"derniere": None, "compte": 0}

    print(f"Ecoute @ {CENTER_FREQ/1e6:.3f} MHz  (Ctrl+C pour arreter)")
    try:
        while True:
            # Lecture bloquante d'un bloc d'echantillons IQ (complexes).
            iq = sdr.read_samples(CHUNK)
            audio = rx.process(np.asarray(iq, dtype=complex))
            frames = decode_audio(audio, FS_AUDIO)
            afficher_trames(frames, vues)
    except KeyboardInterrupt:
        print("\nArret.")
    finally:
        sdr.close()


# =============================================================================
#  MODE AUTOTEST : pas de materiel, on fabrique un signal IQ FM synthetique
#  ----------------------------------------------------------------------------
#  On genere de l'audio AFSK (via kiwi_decoder), on le module en FM pour
#  obtenir un flux IQ realiste, puis on le passe dans EXACTEMENT la meme chaine
#  de reception (FMReceiver + decode_audio) que le mode temps reel.
# =============================================================================
def autotest():
    import wave
    print("=== AUTOTEST kiwi_rt : IQ FM synthetique -> demod -> decode ===")

    # 1) Generer de l'audio AFSK avec de vraies trames (reutilise le module).
    test_frames = [
        {"voies": [10, 50, 100, 150, 200, 254, 0, 128], "alim": 90},
        {"voies": [20, 40, 60, 80, 120, 200, 33, 77],   "alim": 88},
    ]
    kd.make_test_wav("/tmp/kiwi_rt_audio.wav", test_frames, fs=FS_AUDIO, repeat=3)
    with wave.open("/tmp/kiwi_rt_audio.wav", "rb") as w:
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(float) / 32767.0

    # 2) Suréchantillonner l'audio vers FS_IN puis le moduler en FM.
    decim = int(round(FS_IN / FS_AUDIO))
    audio_up = np.repeat(audio, decim)          # suréchantillonnage simple
    kf = 2500.0                                 # excursion de frequence (Hz)
    phase = 2 * np.pi * kf * np.cumsum(audio_up) / FS_IN
    iq = np.exp(1j * phase)                     # signal IQ FM (porteuse a 0 Hz)
    # un peu de bruit pour realisme
    iq += (np.random.normal(0, 0.05, len(iq)) + 1j*np.random.normal(0, 0.05, len(iq)))

    # 3) Passer l'IQ dans la chaine de reception, par blocs (comme en direct).
    rx = FMReceiver(FS_IN, FS_AUDIO, CHAN_BW)
    all_frames = []
    for start in range(0, len(iq), CHUNK):
        bloc = iq[start:start + CHUNK]
        a = rx.process(bloc)
        all_frames += decode_audio(a, FS_AUDIO)

    valid = [f for f in all_frames if f["valide"]]
    print(f"Trames detectees : {len(all_frames)} | valides (checksum) : {len(valid)}")
    for f in valid:
        print(f"  voies_raw={f['voies_raw']}  Vbat~{f['vbat_volts']}V")
    print("OK" if valid else "ECHEC : aucune trame valide")


# =============================================================================
if __name__ == "__main__":
    if "--test" in sys.argv:
        autotest()
    else:
        temps_reel()
