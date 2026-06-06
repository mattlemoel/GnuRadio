#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  kiwi_decoder_868.py
#  Decodeur de telemesure KIKIWI 868 MHz (variante UHF, AFSK Bell-202)
#  -- derive de kiwi_decoder.py, seuls les PARAMETRES AFSK changent --
# -----------------------------------------------------------------------------
#  Auteur      : Matthieu Le Moel
#  Annee       : 2026
#
#  LICENCES (double licence)
#  -------------------------
#  CODE SOURCE :
#    Copyright (C) 2026  Matthieu Le Moel
#
#    Ce programme est un logiciel libre : vous pouvez le redistribuer et/ou
#    le modifier selon les termes de la Licence Publique Generale GNU (GPL)
#    telle que publiee par la Free Software Foundation, soit la version 3 de
#    la Licence, soit (a votre convenance) toute version ulterieure.
#
#    Ce programme est distribue dans l'espoir qu'il sera utile, mais SANS
#    AUCUNE GARANTIE, ni explicite ni implicite, y compris les garanties de
#    COMMERCIALISATION ou D'ADAPTATION A UN USAGE PARTICULIER. Voir la
#    Licence Publique Generale GNU pour plus de details.
#
#    Vous devriez avoir recu une copie de la GPL avec ce programme ; sinon,
#    voir <https://www.gnu.org/licenses/gpl-3.0.html>.
#
#  DOCUMENTATION ET COMMENTAIRES :
#    Mis a disposition sous licence Creative Commons Attribution -
#    Partage dans les Memes Conditions 4.0 International (CC BY-SA 4.0).
#    <https://creativecommons.org/licenses/by-sa/4.0/deed.fr>
#
#  NB : la GPL est la licence adaptee au code ; la CC BY-SA s'applique
#  surtout au texte explicatif / a la documentation. Les deux sont fournies
#  ici a la demande de l'auteur.
# =============================================================================
#
#  PRESENTATION GENERALE
#  ---------------------
#  Ce programme decode le signal AUDIO issu de la demodulation FM (la sortie
#  de GNU Radio ou de rtl_fm), et NON le signal radio brut. La separation des
#  roles est volontaire : la radio (RF + demod FM) est faite en amont, et ce
#  script se concentre sur le traitement bande de base (AFSK -> octets ->
#  trames), bien plus facile a deboguer en Python.
#
#  Chaine de traitement complete :
#
#     [WAV audio AFSK]
#          |  (1) demodulation AFSK : tonalites 900/1500 Hz -> metrique
#          v
#     metrique continue de decision
#          |  (2a) recuperation d'horloge (ouverture d'oeil) -> 1 bit/symbole
#          |  (2b) desentrelacement UART (start/stop) -> octets
#          v
#     flux d'octets
#          |  (3) recherche de l'octet de synchro 0xFF + parsing de trame
#          v
#     trames {8 voies, tension piles, checksum valide ?}
#
#  RAPPEL DU STANDARD KIWI (d'apres la doc CNES / Planete Sciences) :
#    - modulation AFSK : un 0 logique = 900 Hz, un 1 logique = 1500 Hz
#    - debit : 600 bauds (600 bits/seconde)
#    - trame de 11 octets : FF V1 V2 V3 V4 V5 V6 V7 V8 Alim Chk
#         * FF       : octet de synchronisation (jamais produit par une voie)
#         * V1..V8   : les 8 voies de mesure (1 octet chacune)
#         * Alim     : 1/3 de la tension d'alimentation (suivi de l'etat piles)
#         * Chk      : controle = (somme des voies + Alim) puis division par 2
#    - chaque octet circule en UART asynchrone : 1 bit start, 8 bits de
#      donnee (LSB en premier), 1 bit stop.
#
#  AVERTISSEMENT IMPORTANT
#  -----------------------
#  Le format de trame ci-dessus est celui de l'emetteur KIWI (predecesseur).
#  Le KIKIWI actuel numerise sur 10 bits (valeurs 0..1023) et a une pleine
#  echelle de 3,00 V : le format de trame est donc probablement plus large.
#  La polarite de ligne, l'ordre des bits et la presence d'une parite doivent
#  etre CONFIRMES sur une capture reelle. Tous ces parametres sont regroupes
#  dans la section "PARAMETRES AJUSTABLES" ci-dessous pour etre modifies
#  sans toucher au reste du code.
# =============================================================================

import numpy as np               # calcul vectoriel (signaux echantillonnes)
from scipy.signal import lfilter  # filtre RII/RIF : sert ici de filtre adapte
import wave                       # lecture/ecriture de fichiers WAV standard
import sys                        # arguments de ligne de commande


# =============================================================================
#  PARAMETRES AJUSTABLES
#  ---------------------
#  C'est ICI que l'on adapte le decodeur au signal reel (Kiwi vs Kikiwi,
#  conventions UART, etc.). Aucune autre partie du code n'a besoin d'etre
#  modifiee pour un changement de configuration.
# =============================================================================
MARK_HZ    = 1200     # tonalite (Hz) du bit 0 -- Bell-202 (Kikiwi 868 MHz)
SPACE_HZ   = 2200     # tonalite (Hz) du bit 1 -- Bell-202 (Kikiwi 868 MHz)
BAUD       = 1200     # debit symbole (bits/seconde) -- 1200 bauds en 868 MHz
SWAP_TONES = False    # mettre True si les bits sortent inverses (ambiguite mark/space)
FRAME_LEN  = 11       # longueur de trame en octets : FF + V1..V8 + Alim + Chk
SYNC_BYTE  = 0xFF     # octet de synchronisation en tete de trame
FULL_SCALE = 3.0      # pleine echelle en volts : 3,00 V sur Kikiwi (etait 5 V sur Kiwi)
ADC_MAX    = 255      # voir AVERTISSEMENT : le Kikiwi numerise probablement sur 10 bits
                      # (0..1023) -> le format de trame reel est sans doute plus large.
                      # On garde ici la base 8 bits documentee, A CONFIRMER sur capture.

# Conventions de la liaison serie asynchrone (UART) -- A CONFIRMER sur capture
IDLE_LEVEL = 1        # niveau de la ligne au repos (RS232 : repos = 1 = SPACE)
LSB_FIRST  = True     # ordre des bits de donnee (RS232 : bit de poids faible d'abord)


# =============================================================================
#  ETAPE 1 -- DEMODULATION AFSK
#  ----------------------------
#  Objectif : transformer le signal audio (ou alternent deux tonalites) en une
#  "metrique" continue dont le signe indique le bit transmis a chaque instant.
#
#  Methode : filtre adapte non coherent par correlation en quadrature.
#  Pour chaque tonalite, on multiplie le signal par un oscillateur local
#  complexe a la frequence cible, puis on integre l'energie sur la duree d'un
#  symbole. La tonalite reellement presente ressort avec une forte energie,
#  l'autre s'annule. On compare ensuite les deux energies.
# =============================================================================
def afsk_metric(audio, fs):
    """Convertit l'audio AFSK en une metrique de decision continue.

    Parametres
    ----------
    audio : tableau numpy des echantillons audio (normalises)
    fs    : frequence d'echantillonnage du WAV (Hz), p.ex. 48000

    Retour
    ------
    metrique : tableau numpy. >0 => tonalite SPACE (bit 1) dominante ;
               <0 => tonalite MARK (bit 0) dominante.
    """
    # Base de temps : un instant pour chaque echantillon audio.
    n = np.arange(len(audio))
    t = n / fs
    sps = fs / BAUD                       # nombre d'echantillons par symbole

    # Oscillateurs locaux complexes (exponentielles) aux deux frequences.
    # Multiplier le signal par exp(-j.2.pi.f.t) ramene la composante a f
    # autour de 0 Hz, ou l'integration la fait ressortir.
    mark_lo  = np.exp(-1j * 2 * np.pi * MARK_HZ  * t)   # ramene 900 Hz vers 0
    space_lo = np.exp(-1j * 2 * np.pi * SPACE_HZ * t)   # ramene 1500 Hz vers 0

    # Filtre adapte simple : moyenne glissante (boxcar) sur un symbole.
    # lfilter applique ce noyau ; |.|**2 donne l'energie (puissance) de chaque
    # tonalite a chaque instant.
    win = np.ones(int(round(sps))) / sps
    mark_p  = np.abs(lfilter(win, 1.0, audio * mark_lo))  ** 2   # energie a 900 Hz
    space_p = np.abs(lfilter(win, 1.0, audio * space_lo)) ** 2   # energie a 1500 Hz

    # Difference d'energie : le signe donne directement le bit.
    # SWAP_TONES inverse le role des deux tonalites si besoin (calibration).
    metric = space_p - mark_p
    return -metric if SWAP_TONES else metric


# =============================================================================
#  ETAPE 2a -- RECUPERATION D'HORLOGE (SYNCHRONISATION SYMBOLE)
#  -----------------------------------------------------------
#  Le filtre adapte (boxcar) donne une valeur "propre" uniquement lorsqu'on
#  echantillonne en FIN de fenetre symbole (l'integration couvre alors
#  exactement un symbole). On ne connait pas l'instant exact des frontieres
#  symbole : on cherche donc le decalage qui "ouvre le mieux l'oeil", c.-a-d.
#  qui maximise l'amplitude moyenne de la metrique aux instants echantillonnes.
# =============================================================================
def recover_bits(metric, fs):
    """Transforme la metrique continue en une suite de bits (1 bit/symbole).

    On balaie tous les decalages possibles a l'interieur d'un symbole et on
    retient celui qui donne la plus forte amplitude moyenne : c'est l'instant
    d'echantillonnage optimal (oeil le plus ouvert).
    """
    sps = fs / BAUD
    L = int(round(sps))               # periode symbole en echantillons

    # Recherche du meilleur decalage d'echantillonnage 'o' dans [0, L[.
    best_o, best_score = 0, -1.0
    for o in range(L):
        idx = np.arange(o, len(metric), L)        # instants candidats
        score = np.mean(np.abs(metric[idx]))      # ouverture d'oeil moyenne
        if score > best_score:
            best_score, best_o = score, o

    # Echantillonnage final a la cadence symbole, au meilleur decalage.
    idx = np.arange(best_o, len(metric), L)
    bits = (metric[idx] > 0).astype(int)          # 1 = SPACE, 0 = MARK

    # Si la ligne au repos est a 0 (et non 1), on inverse toute la logique.
    if IDLE_LEVEL == 0:
        bits = 1 - bits
    return bits


# =============================================================================
#  ETAPE 2b -- DESENTRELACEMENT UART (TRAME ASYNCHRONE -> OCTETS)
#  -------------------------------------------------------------
#  Chaque octet est encadre facon RS232 : la ligne est au repos (1), un bit
#  de start (0) annonce l'octet, suivent 8 bits de donnee (LSB d'abord par
#  defaut), puis un bit de stop (1). Le bit de stop d'un octet sert de repos
#  a l'octet suivant lorsque les octets s'enchainent sans pause.
# =============================================================================
def bits_to_bytes_uart(bits):
    """Reconstruit des octets a partir d'un flux de bits encode en UART."""
    bytes_out = []
    i, n = 0, len(bits)
    while i < n - 10:                 # il faut au moins 10 bits pour un octet
        # Detection d'un debut d'octet : repos (1) immediatement suivi d'un
        # bit de start (0).
        if bits[i] == 1 and bits[i + 1] == 0:
            data = bits[i + 2:i + 10]      # les 8 bits de donnee
            stop = bits[i + 10]            # le bit de stop attendu (=1)

            # On ne valide l'octet que si le bit de stop est bien present.
            if len(data) == 8 and stop == 1:
                # Reassemblage de l'octet selon l'ordre des bits choisi.
                if LSB_FIRST:
                    val = int(sum(int(b) << k for k, b in enumerate(data)))
                else:
                    val = int(sum(int(b) << (7 - k) for k, b in enumerate(data)))
                bytes_out.append(val)
                i += 10                    # on saute l'octet : son stop = repos suivant
                continue
        i += 1                             # sinon on glisse d'un bit et on reessaie
    return bytes_out


def metric_to_bytes(metric, fs):
    """Raccourci : metrique -> bits -> octets."""
    return bits_to_bytes_uart(recover_bits(metric, fs))


# =============================================================================
#  ETAPE 3 -- PARSING DES TRAMES KIWI
#  ----------------------------------
#  On recherche l'octet de synchro 0xFF, on isole les 11 octets de la trame,
#  on verifie le checksum, puis on convertit les valeurs brutes en tensions.
# =============================================================================
def kiwi_checksum(v1_8, alim):
    """Calcule le checksum Kiwi.

    Regle : addition (sur 8 bits, donc modulo 256) des 8 voies et de la voie
    Alim, suivie d'une division entiere par 2. Le resultat reste <= 127, ce
    qui evite toute confusion avec l'octet de synchro 0xFF.
    """
    s = (sum(v1_8) + alim) & 0xFF     # somme repliee sur 8 bits (modulo 256)
    return s // 2                     # division entiere par 2


def parse_frames(byte_stream):
    """Extrait toutes les trames d'un flux d'octets et evalue leur validite."""
    frames = []
    i = 0
    n = len(byte_stream)
    while i <= n - FRAME_LEN:
        # On se cale sur l'octet de synchronisation.
        if byte_stream[i] == SYNC_BYTE:
            frame = byte_stream[i:i + FRAME_LEN]   # 11 octets a partir du FF
            v = frame[1:9]                         # V1..V8 : les 8 voies
            alim = frame[9]                        # octet Alim (1/3 de la tension)
            chk = frame[10]                        # checksum recu
            ok = (kiwi_checksum(v, alim) == chk)   # comparaison au checksum recalcule

            frames.append({
                "voies_raw":   list(v),            # valeurs brutes 0..ADC_MAX
                # Conversion en volts : valeur / pleine_echelle_numerique * pleine_echelle_volts
                "voies_volts": [round(x / ADC_MAX * FULL_SCALE, 3) for x in v],
                "alim_raw":    alim,
                # Alim represente 1/3 de la tension d'alim : on remultiplie par 3.
                "vbat_volts":  round(alim * 3 / ADC_MAX * FULL_SCALE, 2),
                "chk_recu":    chk,
                "chk_calcule": kiwi_checksum(v, alim),
                "valide":      ok,
            })
            # Si la trame est valide on saute ses 11 octets ; sinon on glisse
            # d'un octet (peut-etre un faux FF a l'interieur des donnees).
            i += FRAME_LEN if ok else 1
        else:
            i += 1
    return frames


def decode_wav(path):
    """Decode un fichier WAV complet : audio -> trames Kiwi.

    Retourne (liste_de_trames, flux_d_octets_brut).
    """
    with wave.open(path, "rb") as w:
        fs = w.getframerate()                       # frequence d'echantillonnage
        raw = w.readframes(w.getnframes())          # tous les echantillons
    # WAV PCM 16 bits mono attendu -> tableau de flottants normalises.
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    audio /= np.max(np.abs(audio)) + 1e-9           # normalisation (+epsilon anti /0)

    metric = afsk_metric(audio, fs)                 # etape 1
    by = metric_to_bytes(metric, fs)                # etapes 2a + 2b
    return parse_frames(by), by                     # etape 3


# =============================================================================
#  OUTIL DE TEST -- GENERATEUR DE SIGNAL SYNTHETIQUE
#  -------------------------------------------------
#  Permet de valider toute la chaine SANS materiel : on fabrique un WAV
#  contenant de vraies trames Kiwi (correctement modulees en AFSK), puis on
#  verifie que le decodeur les retrouve a l'identique. Utile aussi comme banc
#  de test pour ajuster les parametres avant une reception reelle.
# =============================================================================
def uart_bits(byte_val):
    """Encode un octet en bits UART : start(0) + 8 bits de donnee + stop(1)."""
    data = [(byte_val >> k) & 1 for k in range(8)] if LSB_FIRST \
        else [(byte_val >> (7 - k)) & 1 for k in range(8)]
    return [0] + data + [1]


def make_test_wav(path, frames_values, fs=48000, repeat=3, idle_bits=20):
    """Genere un WAV AFSK contenant des trames Kiwi pour tester le decodeur.

    frames_values : liste de dicts {"voies": [8 valeurs], "alim": valeur}.
    repeat        : nombre de repetitions de chaque trame (le Kiwi repete 3x).
    idle_bits     : nombre de bits de repos inseres avant chaque trame.
    """
    sps = fs / BAUD
    all_bits = []

    # Construction de la suite de bits (repos + trames encodees en UART).
    for vals in frames_values:
        v = vals["voies"]
        alim = vals["alim"]
        chk = kiwi_checksum(v, alim)               # checksum coherent
        frame = [SYNC_BYTE] + v + [alim] + [chk]   # trame complete de 11 octets
        for _ in range(repeat):
            all_bits += [IDLE_LEVEL] * idle_bits   # repos avant la trame
            for b in frame:
                all_bits += uart_bits(b)           # chaque octet encadre UART
    all_bits += [IDLE_LEVEL] * idle_bits           # repos final

    # Synthese AFSK : chaque bit devient un segment de sinusoide a sa tonalite.
    # On maintient la continuite de phase pour un signal propre (pas de clics).
    samples = []
    phase = 0.0
    for bit in all_bits:
        f = SPACE_HZ if bit == 1 else MARK_HZ      # 1 -> 1500 Hz, 0 -> 900 Hz
        nsamp = int(round(sps))
        for _ in range(nsamp):
            phase += 2 * np.pi * f / fs            # avancee de phase par echantillon
            samples.append(np.sin(phase))
    sig = np.array(samples)
    sig = (sig * 0.6 * 32767).astype(np.int16)     # mise a l'echelle 16 bits

    # Ecriture du WAV mono 16 bits.
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(sig.tobytes())
    return all_bits


# =============================================================================
#  POINT D'ENTREE
#  --------------
#  - Avec un argument  : decode le WAV fourni (mode reel).
#       python3 kiwi_decoder.py mon_signal.wav
#  - Sans argument     : lance un autotest (genere puis decode un signal).
#       python3 kiwi_decoder.py
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) == 2:
        # ---- MODE REEL : decodage d'un WAV fourni en argument ----
        frames, raw = decode_wav(sys.argv[1])
        print(f"{len(raw)} octets decodes, {len(frames)} trames trouvees")
        for k, f in enumerate(frames):
            tag = "OK " if f["valide"] else "BAD"
            print(f"[{tag}] trame {k}: voies={f['voies_volts']} "
                  f"Vbat~{f['vbat_volts']}V")
    else:
        # ---- MODE AUTOTEST : validation de la chaine sans materiel ----
        print("=== AUTOTEST : generation + decodage d'un signal synthetique ===")
        test_frames = [
            {"voies": [10, 50, 100, 150, 200, 254, 0, 128], "alim": 90},
            {"voies": [20, 40, 60, 80, 120, 200, 33, 77],   "alim": 88},
        ]
        sent = make_test_wav("/tmp/kiwi_test.wav", test_frames)
        frames, raw = decode_wav("/tmp/kiwi_test.wav")
        print(f"Octets decodes : {len(raw)} | Trames trouvees : {len(frames)}")
        ok = 0
        for k, f in enumerate(frames):
            exp = test_frames[k % len(test_frames)]["voies"] if k < 99 else None
            match = (f["voies_raw"] == exp) if exp else "?"
            print(f"  trame {k}: valide={f['valide']} voies_raw={f['voies_raw']} "
                  f"chk {f['chk_recu']}=={f['chk_calcule']} match_attendu={match}")
            if f["valide"]:
                ok += 1
        print(f"\nResultat : {ok}/{len(frames)} trames valides (checksum OK)")
