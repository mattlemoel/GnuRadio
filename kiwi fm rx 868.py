#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  kiwi_fm_rx_868.py
#  Frontal RF GNU Radio pour le KIKIWI 868 MHz (variante UHF)
#  -- derive de kiwi_fm_rx.py : seules la frequence et la largeur canal changent --
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
#  ROLE DE CE PROGRAMME
#  --------------------
#  Capturer le signal VHF de l'emetteur avec un RTL-SDR, le demoduler en FM
#  bande etroite (NBFM), et ecrire un fichier WAV audio a 48 kHz. Ce WAV est
#  ensuite decode par kiwi_decoder.py. La radio s'arrete donc a la sortie
#  "audio" : tout le decodage numerique (AFSK, UART, trames) est fait apres.
#
#  CHAINE DE TRAITEMENT (flowgraph) :
#
#     RTL-SDR  ->  Frequency Xlating FIR  ->  Quadrature Demod (FM)
#              ->  Rational Resampler 48k  ->  Multiply Const  ->  WAV Sink
#
#     * RTL-SDR            : numerise une large bande autour de la porteuse.
#     * Freq Xlating FIR   : recentre + filtre le canal + decime (allege le debit).
#     * Quadrature Demod   : demodulation FM (frequence instantanee -> audio).
#     * Rational Resampler : ramene le debit a 48 kHz (entree du decodeur).
#     * Multiply Const     : mise a l'echelle de l'amplitude avant ecriture.
#     * WAV Sink           : ecrit kiwi_audio.wav (mono, 16 bits, 48 kHz).
#
#  PREREQUIS (installation) :
#     sudo apt install gnuradio gr-osmosdr rtl-sdr
#     # verifier que le dongle est detecte :
#     rtl_test
#
#  UTILISATION :
#     python3 kiwi_fm_rx.py            # enregistre DURATION_S secondes
#     python3 kiwi_decoder.py kiwi_audio.wav
#
#  ---------------------------------------------------------------------------
#  ALTERNATIVE EN UNE LIGNE (sans GNU Radio, avec rtl-sdr + sox) :
#
#     rtl_fm -f 869.500M -M fm -s 48000 -r 48000 -g 40 - \
#         | sox -t raw -r 48000 -e signed -b 16 -c 1 - kiwi_audio.wav
#
#     puis :  python3 kiwi_decoder.py kiwi_audio.wav
#  ---------------------------------------------------------------------------
# =============================================================================

import sys
import time

# Briques GNU Radio. 'gr' est le moteur ; les autres modules fournissent les
# blocs de traitement (sources, filtres, demodulateurs, sinks).
from gnuradio import gr, blocks, filter, analog
from gnuradio.filter import firdes          # calcul des coefficients de filtre

# gr-osmosdr fournit la source materielle pour RTL-SDR (et autres SDR).
try:
    import osmosdr
except ImportError:
    sys.exit("gr-osmosdr manquant : sudo apt install gr-osmosdr")


# =============================================================================
#  PARAMETRES AJUSTABLES
#  ---------------------
#  A adapter selon le materiel et la configuration de l'emetteur.
# =============================================================================
CENTER_FREQ = 137.950e6   # frequence d'emission (ou 138.500e6 selon config)
SAMP_RATE   = 1_024_000   # debit d'echantillonnage du RTL-SDR (Hz)
AUDIO_RATE  = 48_000      # debit audio en sortie (= entree du decodeur)
NBFM_BW     = 15_000      # canal NBFM (Hz) ; canalisation 25 kHz en 868
RF_GAIN     = 40          # gain du tuner (dB) -- a ajuster selon le niveau recu
DURATION_S  = 60          # duree d'enregistrement (secondes)
OUT_WAV     = "kiwi_audio.wav"   # fichier de sortie


# =============================================================================
#  DEFINITION DU FLOWGRAPH
#  -----------------------
#  Un "top_block" GNU Radio contient les blocs et leurs connexions.
# =============================================================================
class KiwiFmRx(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Kiwi FM RX")

        # --- 1) SOURCE RTL-SDR ------------------------------------------------
        # Numerise une bande de largeur SAMP_RATE centree sur CENTER_FREQ.
        self.src = osmosdr.source(args="numchan=1")
        self.src.set_sample_rate(SAMP_RATE)
        self.src.set_center_freq(CENTER_FREQ)
        self.src.set_gain(RF_GAIN)
        self.src.set_freq_corr(0)    # correction de derive (ppm) du dongle -- A REGLER

        # --- 2) SELECTION DE CANAL + DECIMATION ------------------------------
        # On reduit le debit pour ne garder qu'une bande etroite autour du
        # signal, ce qui allege fortement les calculs en aval.
        decim = int(SAMP_RATE // (NBFM_BW * 8))   # facteur de decimation -> ~128 kHz
        chan_rate = SAMP_RATE / decim             # debit apres decimation
        # Filtre passe-bas anti-repliement, calcule pour la largeur NBFM.
        taps = firdes.low_pass(1.0, SAMP_RATE, NBFM_BW, NBFM_BW / 4)
        # Le bloc "freq xlating" recentre (offset 0 ici), filtre et decime.
        self.chan = filter.freq_xlating_fir_filter_ccc(decim, taps, 0, SAMP_RATE)

        # --- 3) DEMODULATION FM ----------------------------------------------
        # La demodulation par quadrature extrait la frequence instantanee, qui
        # constitue le signal audio (ou alternent les tonalites 900/1500 Hz).
        # Le gain convertit l'ecart de frequence en amplitude : il vaut
        # chan_rate / (2*pi*deviation), avec une deviation NBFM ~ 2,5 kHz.
        deviation = 2_500
        self.demod = analog.quadrature_demod_cf(chan_rate / (2 * 3.14159 * deviation))

        # --- 4) REECHANTILLONNAGE VERS 48 kHz --------------------------------
        # Le decodeur attend de l'audio a 48 kHz : on adapte le debit.
        self.resamp = filter.rational_resampler_fff(
            interpolation=int(AUDIO_RATE),
            decimation=int(chan_rate),
        )

        # --- 5) MISE A L'ECHELLE + ECRITURE WAV ------------------------------
        self.scale = blocks.multiply_const_ff(0.8)   # evite la saturation a l'ecriture
        self.sink = blocks.wavfile_sink(
            OUT_WAV, 1, AUDIO_RATE,
            blocks.FORMAT_WAV, blocks.FORMAT_PCM_16,  # WAV mono 16 bits
        )

        # --- CONNEXION DES BLOCS DANS L'ORDRE DE LA CHAINE -------------------
        self.connect(self.src, self.chan, self.demod,
                     self.resamp, self.scale, self.sink)


# =============================================================================
#  POINT D'ENTREE
#  --------------
#  Demarre le flowgraph, enregistre pendant DURATION_S, puis s'arrete
#  proprement. (Ctrl+C interrompt egalement l'enregistrement.)
# =============================================================================
if __name__ == "__main__":
    tb = KiwiFmRx()
    print(f"Enregistrement {DURATION_S}s @ {CENTER_FREQ/1e6:.3f} MHz -> {OUT_WAV}")
    tb.start()                       # lance le traitement en arriere-plan
    try:
        time.sleep(DURATION_S)       # laisse tourner le temps voulu
    except KeyboardInterrupt:
        pass                         # arret anticipe par l'utilisateur
    tb.stop()                        # demande l'arret du flowgraph
    tb.wait()                        # attend la fin propre des traitements
    print(f"Termine. Decode avec :  python3 kiwi_decoder.py {OUT_WAV}")
