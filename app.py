"""
================================================================================
HYBRID MECLIS CONTROLLER
--------------------------------------------------------------------------------
ABD Kongresi Yasa Oylama Olasiligi Hibrit Karar Destek Simulatoru.

Mimari:
    1) Bulanik Mantik Motoru (Mamdani Cikarimi - scikit-fuzzy)
       Girdi (Antecedent) : Lobi Gucu [0-100], Partizan Kutuplasma [0-10]
       Cikti (Consequent) : Ham Destek Orani [0-100]
       Durulastirma       : Centroid

    2) Sembolik Matematik Motoru (SymPy)
       Dissal makro faktorler (Savas/Rally Effect, Medya Skandali,
       Yuksek Enflasyon) if-else ile DEGIL, cebirsel bir polinom
       denklemi ile modellenir ve sp.lambdify ile numerik bir
       fonksiyona donusturulur.

    3) Monte Carlo Simulasyonu
       Nihai olasilik etrafinda 10.000 iterasyonluk stokastik dagilim.

Ders: Benzetim Programlari (Simulasyon) - Akademik Prototip
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import streamlit as st


# ==============================================================================
# 1. VERI YAPILARI (DATACLASSES)
# ==============================================================================

@dataclass
class SenaryoGirdileri:
    """Kullanicinin arayuzden sectigi senaryo parametrelerini tutan veri sinifi."""
    lobi_gucu: float           # 0-100 arasi
    partizan_kutuplasma: float  # 0-10 arasi
    savas_durumu: bool          # Rally-Around-the-Flag etkisi acik mi?
    medya_skandali: bool        # Medya skandali etkisi acik mi?
    yuksek_enflasyon: bool      # Yuksek enflasyon etkisi acik mi?
    yasayi_sunan_parti: str = "Cumhuriyetci"  # Coklu ajan modelinde parti sadakati icin


@dataclass
class SimulasyonSonucu:
    """Hibrit motorun urettigi tum ciktilarin tutuldugu veri sinifi."""
    fuzzy_ham_skor: float
    sembolik_nihai_skor: float
    nihai_olasilik: float
    monte_carlo_dagilimi: np.ndarray = field(repr=False)


@dataclass
class AjanGrubuSonucu:
    """
    Coklu ajan (435 temsilci) simulasyonunun ciktisini tutan veri sinifi.
    Her ajan, kendi lobi/kutuplasma maruziyetine gore bagimsiz olarak
    fuzzy + sembolik boru hattindan gecirilir.
    """
    parti: np.ndarray = field(repr=False)
    lobi_maruziyeti: np.ndarray = field(repr=False)
    kutuplasma_maruziyeti: np.ndarray = field(repr=False)
    nihai_skor: np.ndarray = field(repr=False)
    parti_sadakati_etkisi: np.ndarray = field(repr=False)
    oy_evet: np.ndarray = field(repr=False)  # boolean dizi
    yasayi_sunan_parti: str
    evet_sayisi: int
    hayir_sayisi: int
    toplam_ajan: int

    @property
    def kabul_edildi_mi(self) -> bool:
        """Basit cogunluk kurali (yarisi + 1)."""
        return self.evet_sayisi > self.toplam_ajan / 2


# ==============================================================================
# 2. HIBRIT MECLIS CONTROLLER SINIFI
# ==============================================================================

class HybridMeclisController:
    """
    Bulanik Mantik (scikit-fuzzy) ve Sembolik Matematik (SymPy) motorlarini
    birlestiren hibrit karar destek denetleyicisi.

    Akis:
        SenaryoGirdileri -> [Fuzzy Mamdani Motoru] -> Ham Skor (F)
                          -> [SymPy Polinom Motoru] -> Nihai Skor (P)
                          -> [Monte Carlo]           -> Olasilik Dagilimi
    """

    def __init__(self) -> None:
        self._lobi: ctrl.Antecedent
        self._kutuplasma: ctrl.Antecedent
        self._destek: ctrl.Consequent
        self._fuzzy_sistem: ctrl.ControlSystem
        self._sembolik_fonksiyon: Callable[[float, float, float, float], float]

        self._fuzzy_degiskenleri_kur()
        self._fuzzy_kurallari_kur()
        self._sembolik_modeli_kur()

    # --------------------------------------------------------------------
    # 2.1 BULANIK MANTIK MOTORU (MAMDANI)
    # --------------------------------------------------------------------
    def _fuzzy_degiskenleri_kur(self) -> None:
        """Antecedent / Consequent uzaylarini ve uyelik fonksiyonlarini tanimlar."""

        # Universe of discourse (evren kumeleri)
        lobi_uzayi = np.arange(0, 101, 1)
        kutuplasma_uzayi = np.arange(0, 11, 0.1)
        destek_uzayi = np.arange(0, 101, 1)

        self._lobi = ctrl.Antecedent(lobi_uzayi, "lobi_gucu")
        self._kutuplasma = ctrl.Antecedent(kutuplasma_uzayi, "partizan_kutuplasma")
        self._destek = ctrl.Consequent(destek_uzayi, "ham_destek")

        # --- Lobi Gucu uyelik fonksiyonlari (trapmf ile kenarlar, trimf ile orta) ---
        self._lobi["dusuk"] = fuzz.trapmf(lobi_uzayi, [0, 0, 15, 40])
        self._lobi["orta"] = fuzz.trimf(lobi_uzayi, [20, 50, 80])
        self._lobi["yuksek"] = fuzz.trapmf(lobi_uzayi, [60, 85, 100, 100])

        # --- Partizan Kutuplasma uyelik fonksiyonlari ---
        self._kutuplasma["dusuk"] = fuzz.trapmf(kutuplasma_uzayi, [0, 0, 1.5, 4])
        self._kutuplasma["orta"] = fuzz.trimf(kutuplasma_uzayi, [2, 5, 8])
        self._kutuplasma["yuksek"] = fuzz.trapmf(kutuplasma_uzayi, [6, 8.5, 10, 10])

        # --- Ham Destek Orani (Consequent) uyelik fonksiyonlari ---
        self._destek["zayif"] = fuzz.trapmf(destek_uzayi, [0, 0, 15, 40])
        self._destek["kararsiz"] = fuzz.trimf(destek_uzayi, [25, 50, 75])
        self._destek["kabul_edilebilir"] = fuzz.trimf(destek_uzayi, [55, 75, 90])
        self._destek["guclu"] = fuzz.trapmf(destek_uzayi, [80, 92, 100, 100])

    def _fuzzy_kurallari_kur(self) -> None:
        """IF-THEN bicimindeki bulanik mantik kural tabanini olusturur (>= 5 kural)."""

        kural_1 = ctrl.Rule(
            self._lobi["yuksek"] & self._kutuplasma["dusuk"],
            self._destek["guclu"],
        )
        kural_2 = ctrl.Rule(
            self._lobi["yuksek"] & self._kutuplasma["orta"],
            self._destek["kabul_edilebilir"],
        )
        kural_3 = ctrl.Rule(
            self._lobi["yuksek"] & self._kutuplasma["yuksek"],
            self._destek["kararsiz"],
        )
        kural_4 = ctrl.Rule(
            self._lobi["orta"] & self._kutuplasma["dusuk"],
            self._destek["kabul_edilebilir"],
        )
        kural_5 = ctrl.Rule(
            self._lobi["orta"] & self._kutuplasma["yuksek"],
            self._destek["zayif"],
        )
        kural_6 = ctrl.Rule(
            self._lobi["dusuk"],
            self._destek["zayif"],
        )
        kural_7 = ctrl.Rule(
            self._kutuplasma["yuksek"] & ~self._lobi["yuksek"],
            self._destek["zayif"],
        )

        self._fuzzy_sistem = ctrl.ControlSystem(
            [kural_1, kural_2, kural_3, kural_4, kural_5, kural_6, kural_7]
        )

    def fuzzy_skor_hesapla(self, lobi_gucu: float, kutuplasma: float) -> float:
        """
        Mamdani cikarimini calistirir ve Centroid durulastirma yontemi ile
        ham (defuzzified) destek skorunu dondurur.
        """
        simulasyon = ctrl.ControlSystemSimulation(self._fuzzy_sistem)
        simulasyon.input["lobi_gucu"] = float(np.clip(lobi_gucu, 0, 100))
        simulasyon.input["partizan_kutuplasma"] = float(np.clip(kutuplasma, 0, 10))
        simulasyon.compute()
        return float(simulasyon.output["ham_destek"])

    # --------------------------------------------------------------------
    # 2.2 SEMBOLIK MATEMATIK MOTORU (SYMPY)
    # --------------------------------------------------------------------
    def _sembolik_modeli_kur(self) -> None:
        """
        Dissal makro faktorleri if-else yerine cebirsel bir polinom
        denklemiyle modeller:

            P = F + 15*W - 20*Sc - 10*Inf

            F   : Fuzzy motorundan gelen ham destek skoru
            W   : Savas durumu (Rally Effect) carpani  (0 veya 1)
            Sc  : Medya skandali carpani                (0 veya 1)
            Inf : Yuksek enflasyon carpani               (0 veya 1)

        Denklem sp.lambdify ile yuksek performansli numerik bir
        fonksiyona donusturulur.
        """
        F, W, Sc, Inf = sp.symbols("F W Sc Inf", real=True)

        self._sembolik_semboller = (F, W, Sc, Inf)
        self._sembolik_denklem: sp.Expr = F + 15 * W - 20 * Sc - 10 * Inf

        self._sembolik_fonksiyon = sp.lambdify(
            (F, W, Sc, Inf), self._sembolik_denklem, modules="numpy"
        )

    def sembolik_denklemi_goster(self) -> str:
        """Kurulan cebirsel denklemin LaTeX gosterimini dondurur (arayuzde gosterim icin)."""
        return sp.latex(sp.Eq(sp.Symbol("P"), self._sembolik_denklem))

    def nihai_skor_hesapla(
        self,
        ham_skor: float,
        savas_durumu: bool,
        medya_skandali: bool,
        yuksek_enflasyon: bool,
    ) -> float:
        """Lambdify edilmis sembolik fonksiyonu numerik olarak degerlendirir."""
        sonuc = self._sembolik_fonksiyon(
            ham_skor,
            1.0 if savas_durumu else 0.0,
            1.0 if medya_skandali else 0.0,
            1.0 if yuksek_enflasyon else 0.0,
        )
        return float(np.clip(sonuc, 0, 100))

    # --------------------------------------------------------------------
    # 2.3 MONTE CARLO SIMULASYONU
    # --------------------------------------------------------------------
    @staticmethod
    def monte_carlo_calistir(
        nihai_olasilik: float, iterasyon: int = 10_000, std_sapma: float = 7.5
    ) -> np.ndarray:
        """
        Nihai olasiligi merkez (mu) kabul ederek, normal dagilimli
        10.000 iterasyonluk bir stokastik benzetim (can egrisi) uretir.
        """
        rng = np.random.default_rng()
        dagilim = rng.normal(loc=nihai_olasilik, scale=std_sapma, size=iterasyon)
        return np.clip(dagilim, 0, 100)

    # --------------------------------------------------------------------
    # 2.4 UCTAN UCA CALISTIRMA (ORKESTRASYON)
    # --------------------------------------------------------------------
    def simulasyonu_calistir(self, girdi: SenaryoGirdileri) -> SimulasyonSonucu:
        """Tum hibrit boru hattini (pipeline) calistirip nihai sonucu dondurur."""

        ham_skor = self.fuzzy_skor_hesapla(girdi.lobi_gucu, girdi.partizan_kutuplasma)

        nihai_skor = self.nihai_skor_hesapla(
            ham_skor,
            girdi.savas_durumu,
            girdi.medya_skandali,
            girdi.yuksek_enflasyon,
        )

        mc_dagilimi = self.monte_carlo_calistir(nihai_skor)

        return SimulasyonSonucu(
            fuzzy_ham_skor=ham_skor,
            sembolik_nihai_skor=nihai_skor,
            nihai_olasilik=nihai_skor,
            monte_carlo_dagilimi=mc_dagilimi,
        )

    # --------------------------------------------------------------------
    # 2.5 GORSELLESTIRME YARDIMCILARI
    # --------------------------------------------------------------------
    def uyelik_fonksiyonlari_grafigi(self) -> plt.Figure:
        """scikit-fuzzy uyelik fonksiyonlarini matplotlib ile cizer."""
        fig, eksenler = plt.subplots(nrows=3, figsize=(8, 7))

        for etiket in self._lobi.terms:
            eksenler[0].plot(
                self._lobi.universe,
                self._lobi[etiket].mf,
                label=etiket,
                linewidth=2,
            )
        eksenler[0].set_title("Lobi Gucu Uyelik Fonksiyonlari")
        eksenler[0].legend(loc="upper right")

        for etiket in self._kutuplasma.terms:
            eksenler[1].plot(
                self._kutuplasma.universe,
                self._kutuplasma[etiket].mf,
                label=etiket,
                linewidth=2,
            )
        eksenler[1].set_title("Partizan Kutuplasma Uyelik Fonksiyonlari")
        eksenler[1].legend(loc="upper right")

        for etiket in self._destek.terms:
            eksenler[2].plot(
                self._destek.universe,
                self._destek[etiket].mf,
                label=etiket,
                linewidth=2,
            )
        eksenler[2].set_title("Ham Destek Orani Uyelik Fonksiyonlari (Consequent)")
        eksenler[2].legend(loc="upper right")

        fig.tight_layout()
        return fig

    @staticmethod
    def monte_carlo_histogrami(dagilim: np.ndarray, nihai_olasilik: float) -> go.Figure:
        """Monte Carlo dagilimini plotly ile interaktif histogram (can egrisi) olarak cizer."""
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=dagilim,
                nbinsx=60,
                marker_color="rgba(59, 130, 246, 0.75)",
                name="Monte Carlo Iterasyonlari",
            )
        )
        fig.add_vline(
            x=float(np.mean(dagilim)),
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Ortalama: {np.mean(dagilim):.1f}",
            annotation_position="top",
        )
        fig.update_layout(
            title=f"Monte Carlo Olasilik Dagilimi (n=10.000) — Nihai Skor: {nihai_olasilik:.2f}",
            xaxis_title="Kabul Olasiligi (%)",
            yaxis_title="Frekans",
            bargap=0.02,
            template="plotly_white",
        )
        return fig

    # --------------------------------------------------------------------
    # 2.6 DUYARLILIK ANALIZI (SENSITIVITY ANALYSIS)
    # --------------------------------------------------------------------
    def duyarlilik_analizi_calistir(
        self, girdi: SenaryoGirdileri, oran: float = 0.15
    ) -> dict[str, float]:
        """
        Her parametreyi tek tek perturbe ederek nihai skorun ne kadar
        degistigini olcer (tornado diyagrami icin ham veri).

        Surekli degiskenler (lobi, kutuplasma) icin +/- `oran` kadar
        oynatilir; ikili (boolean) makro faktorler icin acik/kapali
        farki olcek olarak kullanilir.
        """

        def skor_hesapla(g: SenaryoGirdileri) -> float:
            ham = self.fuzzy_skor_hesapla(g.lobi_gucu, g.partizan_kutuplasma)
            return self.nihai_skor_hesapla(
                ham, g.savas_durumu, g.medya_skandali, g.yuksek_enflasyon
            )

        etkiler: dict[str, float] = {}

        # --- Lobi Gucu: +-%oran ---
        lobi_yuksek = SenaryoGirdileri(**{**girdi.__dict__, "lobi_gucu": min(girdi.lobi_gucu * (1 + oran), 100)})
        lobi_dusuk = SenaryoGirdileri(**{**girdi.__dict__, "lobi_gucu": max(girdi.lobi_gucu * (1 - oran), 0)})
        etkiler["Lobi Gucu"] = skor_hesapla(lobi_yuksek) - skor_hesapla(lobi_dusuk)

        # --- Kutuplasma: +-%oran ---
        kutup_yuksek = SenaryoGirdileri(**{**girdi.__dict__, "partizan_kutuplasma": min(girdi.partizan_kutuplasma * (1 + oran), 10)})
        kutup_dusuk = SenaryoGirdileri(**{**girdi.__dict__, "partizan_kutuplasma": max(girdi.partizan_kutuplasma * (1 - oran), 0)})
        etkiler["Partizan Kutuplasma"] = skor_hesapla(kutup_yuksek) - skor_hesapla(kutup_dusuk)

        # --- Boolean makro faktorler: acik (True) vs kapali (False) farki ---
        for alan, etiket in [
            ("savas_durumu", "Savas Durumu (Rally)"),
            ("medya_skandali", "Medya Skandali"),
            ("yuksek_enflasyon", "Yuksek Enflasyon"),
        ]:
            acik = SenaryoGirdileri(**{**girdi.__dict__, alan: True})
            kapali = SenaryoGirdileri(**{**girdi.__dict__, alan: False})
            etkiler[etiket] = skor_hesapla(acik) - skor_hesapla(kapali)

        return etkiler

    @staticmethod
    def duyarlilik_tornado_grafigi(etkiler: dict[str, float]) -> go.Figure:
        """Duyarlilik analizi sonuclarini buyukluge gore siralanmis tornado bar grafigi olarak cizer."""
        siralanmis = dict(sorted(etkiler.items(), key=lambda kv: abs(kv[1])))
        etiketler = list(siralanmis.keys())
        degerler = list(siralanmis.values())
        renkler = ["#16a34a" if v >= 0 else "#dc2626" for v in degerler]

        fig = go.Figure(
            go.Bar(
                x=degerler,
                y=etiketler,
                orientation="h",
                marker_color=renkler,
                text=[f"{v:+.2f}" for v in degerler],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Duyarlilik Analizi — Parametrelerin Nihai Skora Etkisi",
            xaxis_title="Nihai Skordaki Degisim (Δ)",
            template="plotly_white",
            height=350,
        )
        fig.add_vline(x=0, line_color="gray", line_width=1)
        return fig

    # --------------------------------------------------------------------
    # 2.7 COKLU AJAN MODELI (AGENT-BASED MECLIS SIMULASYONU)
    # --------------------------------------------------------------------
    def _fuzzy_skor_vektorel(
        self, lobi_array: np.ndarray, kutuplasma_array: np.ndarray
    ) -> np.ndarray:
        """
        Yuzlerce ajan icin performansli vektorize Mamdani cikarimi.
        ControlSystemSimulation her ajan icin ayri ayri kurulmadigindan
        (maliyetli oldugu icin), uyelik dereceleri `fuzz.interp_membership`
        ile dizi (array) bazinda hesaplanir; durulastirma asamasinda ise
        her cikti teriminin kendi centroid agirlik merkezi kullanilarak
        agirlikli ortalama (Sugeno-stili hizli defuzzifikasyon) uygulanir.
        Bu, klasik Mamdani-centroid yontemine performans/dogruluk
        dengesi gozetilerek getirilen vektorize bir yaklasiklamadir.
        """
        lobi_u = self._lobi.universe
        kutup_u = self._kutuplasma.universe
        destek_u = self._destek.universe

        lobi_dusuk = fuzz.interp_membership(lobi_u, self._lobi["dusuk"].mf, lobi_array)
        lobi_orta = fuzz.interp_membership(lobi_u, self._lobi["orta"].mf, lobi_array)
        lobi_yuksek = fuzz.interp_membership(lobi_u, self._lobi["yuksek"].mf, lobi_array)

        kutup_dusuk = fuzz.interp_membership(kutup_u, self._kutuplasma["dusuk"].mf, kutuplasma_array)
        kutup_orta = fuzz.interp_membership(kutup_u, self._kutuplasma["orta"].mf, kutuplasma_array)
        kutup_yuksek = fuzz.interp_membership(kutup_u, self._kutuplasma["yuksek"].mf, kutuplasma_array)

        # Kural atesleme gucleri (AND = min) — kurallar fuzzy_kurallari_kur ile birebir esler
        r1 = np.minimum(lobi_yuksek, kutup_dusuk)                  # -> guclu
        r2 = np.minimum(lobi_yuksek, kutup_orta)                   # -> kabul_edilebilir
        r3 = np.minimum(lobi_yuksek, kutup_yuksek)                 # -> kararsiz
        r4 = np.minimum(lobi_orta, kutup_dusuk)                    # -> kabul_edilebilir
        r5 = np.minimum(lobi_orta, kutup_yuksek)                   # -> zayif
        r6 = lobi_dusuk                                            # -> zayif
        r7 = np.minimum(kutup_yuksek, 1.0 - lobi_yuksek)           # -> zayif

        aktivasyon_guclu = r1
        aktivasyon_kabul = np.maximum(r2, r4)
        aktivasyon_kararsiz = r3
        aktivasyon_zayif = np.maximum.reduce([r5, r6, r7])

        merkez_zayif = fuzz.defuzz(destek_u, self._destek["zayif"].mf, "centroid")
        merkez_kararsiz = fuzz.defuzz(destek_u, self._destek["kararsiz"].mf, "centroid")
        merkez_kabul = fuzz.defuzz(destek_u, self._destek["kabul_edilebilir"].mf, "centroid")
        merkez_guclu = fuzz.defuzz(destek_u, self._destek["guclu"].mf, "centroid")

        pay = (
            aktivasyon_zayif * merkez_zayif
            + aktivasyon_kararsiz * merkez_kararsiz
            + aktivasyon_kabul * merkez_kabul
            + aktivasyon_guclu * merkez_guclu
        )
        payda = aktivasyon_zayif + aktivasyon_kararsiz + aktivasyon_kabul + aktivasyon_guclu
        payda = np.where(payda <= 1e-9, 1e-9, payda)
        return pay / payda

    def coklu_ajan_simulasyonu_calistir(
        self,
        girdi: SenaryoGirdileri,
        ajan_sayisi: int = 435,
        cumhuriyetci_orani: float = 0.51,
        tohum: int | None = None,
    ) -> AjanGrubuSonucu:
        """
        435 sanal temsilciyi (Cumhuriyetci/Demokrat) bagimsiz ajanlar olarak
        modelleyip, her birinin kendi lobi ve kutuplasma maruziyetine gore
        hibrit fuzzy+sembolik boru hattindan gecirir ve gercek bir
        "oylama sayimi" (roll-call) simule eder.

        PARTI SADAKATI (Party-Line Voting):
        Gercek Kongre'de oylama sadece bireysel lobi/kutuplasma maruziyetine
        gore degil, buyuk olcude "yasayi kim sundu" sorusuna gore sekillenir.
        Bu nedenle her ajanin nihai skoruna, kendi partisinin yasayi sunup
        sunmadigina bagli bir parti sadakati terimi eklenir. Bu terimin
        buyuklugu kutuplasma seviyesiyle olceklenir: kutuplasma dusukken
        ajanlar nispeten bagimsiz oy kullanir, kutuplasma yuksekken parti
        disiplini sertlesir ve oylar buyuk olcude parti cizgisinde toplanir.
        Bu, if-else degil; parti uyumunu +1/-1 olarak kodlayan vektorize
        bir cebirsel terimdir (TABAN_SADAKAT + KUTUPLASMA_KATSAYISI * kutuplasma).
        """
        rng = np.random.default_rng(tohum)

        n_cumhuriyetci = int(round(ajan_sayisi * cumhuriyetci_orani))
        n_demokrat = ajan_sayisi - n_cumhuriyetci
        parti = np.array(["Cumhuriyetci"] * n_cumhuriyetci + ["Demokrat"] * n_demokrat)

        # Parti tarafsiz bir taban dagilim: her ajanin maruziyeti, kullanicinin
        # sectigi senaryo merkez (mu) kabul edilerek normal dagilimla cesitlendirilir.
        lobi_array = np.clip(rng.normal(loc=girdi.lobi_gucu, scale=14.0, size=ajan_sayisi), 0, 100)
        kutuplasma_array = np.clip(
            rng.normal(loc=girdi.partizan_kutuplasma, scale=1.3, size=ajan_sayisi), 0, 10
        )

        ham_skorlar = self._fuzzy_skor_vektorel(lobi_array, kutuplasma_array)

        nihai_skorlar_makro = self._sembolik_fonksiyon(
            ham_skorlar,
            1.0 if girdi.savas_durumu else 0.0,
            1.0 if girdi.medya_skandali else 0.0,
            1.0 if girdi.yuksek_enflasyon else 0.0,
        )

        # --- Parti Sadakati Terimi (vektorize, if-else degil) ---
        TABAN_SADAKAT = 18.0          # Dusuk kutuplasmada bile var olan asgari parti disiplini
        KUTUPLASMA_KATSAYISI = 3.0    # Kutuplasma arttikca parti cizgisi sertlesir (maks +30 @ kutuplasma=10)

        parti_uyumu = np.where(parti == girdi.yasayi_sunan_parti, 1.0, -1.0)
        parti_sadakati_etkisi = parti_uyumu * (TABAN_SADAKAT + KUTUPLASMA_KATSAYISI * kutuplasma_array)

        nihai_skorlar = np.clip(nihai_skorlar_makro + parti_sadakati_etkisi, 0, 100)

        oy_olasiligi = nihai_skorlar / 100.0
        oy_evet = rng.random(ajan_sayisi) < oy_olasiligi

        return AjanGrubuSonucu(
            parti=parti,
            lobi_maruziyeti=lobi_array,
            kutuplasma_maruziyeti=kutuplasma_array,
            nihai_skor=nihai_skorlar,
            parti_sadakati_etkisi=parti_sadakati_etkisi,
            oy_evet=oy_evet,
            yasayi_sunan_parti=girdi.yasayi_sunan_parti,
            evet_sayisi=int(np.sum(oy_evet)),
            hayir_sayisi=int(np.sum(~oy_evet)),
            toplam_ajan=ajan_sayisi,
        )

    @staticmethod
    def meclis_oturma_plani_grafigi(sonuc: AjanGrubuSonucu) -> go.Figure:
        """
        435 ajani yari-dairesel bir 'meclis oturma plani' (parliament chart)
        seklinde, oy yonune ve partiye gore renklendirerek cizer.
        """
        n = sonuc.toplam_ajan
        # Yari-daire uzerinde yaklasik esit araliklarla, sira sira yerlesim
        satir_sayisi = 8
        x_list, y_list = [], []
        kalan = n
        for satir in range(satir_sayisi):
            yaricap = 3 + satir * 1.05
            bu_satirdaki = max(1, round(n * (yaricap) / sum(3 + s * 1.05 for s in range(satir_sayisi))))
            bu_satirdaki = min(bu_satirdaki, kalan) if satir < satir_sayisi - 1 else kalan
            acilar = np.linspace(0.12, np.pi - 0.12, bu_satirdaki)
            for aci in acilar:
                x_list.append(yaricap * np.cos(aci))
                y_list.append(yaricap * np.sin(aci))
            kalan -= bu_satirdaki
            if kalan <= 0:
                break

        x_arr = np.array(x_list[:n])
        y_arr = np.array(y_list[:n])

        renkler = np.where(sonuc.oy_evet, "#16a34a", "#dc2626")
        semboller = np.where(sonuc.parti == "Cumhuriyetci", "circle", "diamond")

        fig = go.Figure()
        for parti_adi, sembol in [("Cumhuriyetci", "circle"), ("Demokrat", "diamond")]:
            maske = sonuc.parti == parti_adi
            fig.add_trace(
                go.Scatter(
                    x=x_arr[maske],
                    y=y_arr[maske],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=renkler[maske],
                        symbol=sembol,
                        line=dict(width=0.5, color="white"),
                    ),
                    name=parti_adi,
                    hovertext=[
                        f"{parti_adi} | Skor: {s:.1f} | Oy: {'EVET' if e else 'HAYIR'}"
                        for s, e in zip(sonuc.nihai_skor[maske], sonuc.oy_evet[maske])
                    ],
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            title=f"Meclis Oturma Plani — {sonuc.evet_sayisi} Evet / {sonuc.hayir_sayisi} Hayir "
                  f"({sonuc.toplam_ajan} Temsilci)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            template="plotly_white",
            height=420,
            showlegend=True,
        )
        return fig


# ==============================================================================
# 3. STREAMLIT ARAYUZU
# ==============================================================================

def arayuzu_calistir() -> None:
    st.set_page_config(
        page_title="Hibrit Meclis Oylama Simulatoru",
        page_icon="🏛️",
        layout="wide",
    )

    st.title("🏛️ Hibrit Meclis Oylama Karar Destek Simulatoru")
    st.caption(
        "Bulanik Mantik (Mamdani) + Sembolik Matematik (SymPy) tabanli hibrit model "
        "— Benzetim Programlari Ders Projesi"
    )

    # Controller'i sadece bir kez olustur (Streamlit yeniden calistirmalarinda kalici tut)
    if "controller" not in st.session_state:
        st.session_state.controller = HybridMeclisController()
    controller: HybridMeclisController = st.session_state.controller

    # --- Kenar Cubugu: Senaryo Girdileri ---
    st.sidebar.header("⚙️ Senaryo Parametreleri")

    lobi_gucu = st.sidebar.slider("Lobi Gucu (0-100)", 0, 100, 65)
    kutuplasma = st.sidebar.slider("Partizan Kutuplasma (0-10)", 0.0, 10.0, 4.5, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Dissal Makro Kriz Senaryolari (SymPy)")
    savas_durumu = st.sidebar.toggle("Savas Durumu / Rally Effect", value=False)
    medya_skandali = st.sidebar.toggle("Medya Skandali", value=False)
    yuksek_enflasyon = st.sidebar.toggle("Yuksek Enflasyon", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏛️ Coklu Ajan Modeli")
    yasayi_sunan_parti = st.sidebar.selectbox(
        "Yasayi Sunan Parti", ["Cumhuriyetci", "Demokrat"], index=0
    )

    girdi = SenaryoGirdileri(
        lobi_gucu=lobi_gucu,
        partizan_kutuplasma=kutuplasma,
        savas_durumu=savas_durumu,
        medya_skandali=medya_skandali,
        yuksek_enflasyon=yuksek_enflasyon,
        yasayi_sunan_parti=yasayi_sunan_parti,
    )

    sonuc = controller.simulasyonu_calistir(girdi)

    sekme_ana, sekme_duyarlilik, sekme_ajan = st.tabs(
        ["📊 Ana Simulasyon", "📈 Duyarlilik Analizi", "🏛️ Coklu Ajan Modeli (435 Temsilci)"]
    )

    # ==================================================================
    # SEKME 1: ANA SIMULASYON
    # ==================================================================
    with sekme_ana:
        st.subheader("📊 Anlik Hesaplama Sonuclari")
        metrik_1, metrik_2, metrik_3 = st.columns(3)

        metrik_1.metric(
            label="Fuzzy Ham Skor (Centroid)",
            value=f"{sonuc.fuzzy_ham_skor:.2f}",
        )
        metrik_2.metric(
            label="Nihai SymPy Skoru (P)",
            value=f"{sonuc.sembolik_nihai_skor:.2f}",
            delta=f"{sonuc.sembolik_nihai_skor - sonuc.fuzzy_ham_skor:+.2f} (makro etki)",
        )
        metrik_3.metric(
            label="Nihai Kabul Olasiligi",
            value=f"%{sonuc.nihai_olasilik:.1f}",
        )

        with st.expander("🧮 Sembolik (SymPy) Denklemi Goruntule"):
            st.latex(controller.sembolik_denklemi_goster())
            st.markdown(
                """
                **Degisken Aciklamalari:**
                - `F`   : Bulanik mantik motorundan gelen ham destek skoru (Centroid)
                - `W`   : Savas durumu (Rally-Around-the-Flag) carpani
                - `Sc`  : Medya skandali carpani
                - `Inf` : Yuksek enflasyon carpani

                Bu denklem `sympy.lambdify` ile derlenerek if-else mantigi
                kullanilmadan, sürekli/cebirsel bir fonksiyon olarak hesaplanir.
                """
            )

        st.divider()

        st.subheader("🔣 Bulanik Mantik Uyelik Fonksiyonlari (scikit-fuzzy)")
        fig_uyelik = controller.uyelik_fonksiyonlari_grafigi()
        st.pyplot(fig_uyelik)

        st.divider()

        st.subheader("🎲 Monte Carlo Simulasyonu (10.000 Iterasyon)")
        fig_mc = controller.monte_carlo_histogrami(
            sonuc.monte_carlo_dagilimi, sonuc.nihai_olasilik
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        mc_kol1, mc_kol2, mc_kol3 = st.columns(3)
        mc_kol1.metric("MC Ortalama", f"{np.mean(sonuc.monte_carlo_dagilimi):.2f}")
        mc_kol2.metric("MC Std. Sapma", f"{np.std(sonuc.monte_carlo_dagilimi):.2f}")
        mc_kol3.metric(
            "P(Kabul > 50)",
            f"%{(np.mean(sonuc.monte_carlo_dagilimi > 50) * 100):.1f}",
        )

    # ==================================================================
    # SEKME 2: DUYARLILIK ANALIZI
    # ==================================================================
    with sekme_duyarlilik:
        st.subheader("📈 Duyarlilik Analizi (Tornado Diyagrami)")
        st.markdown(
            """
            Mevcut senaryo etrafinda her parametre tek tek perturbe edilerek
            (surekli degiskenler icin **±%15**, makro faktorler icin **acik/kapali**
            farki) nihai skorun ne kadar degistigi olculur. Cubuk ne kadar uzunsa,
            model o parametreye o kadar **duyarli**dir.
            """
        )

        oran = st.slider("Perturbasyon Orani (surekli degiskenler icin)", 0.05, 0.30, 0.15, step=0.05)
        etkiler = controller.duyarlilik_analizi_calistir(girdi, oran=oran)
        fig_tornado = controller.duyarlilik_tornado_grafigi(etkiler)
        st.plotly_chart(fig_tornado, use_container_width=True)

        en_etkili = max(etkiler, key=lambda k: abs(etkiler[k]))
        st.info(
            f"Bu senaryoda nihai skoru en cok etkileyen parametre: **{en_etkili}** "
            f"(Δ = {etkiler[en_etkili]:+.2f})"
        )

        with st.expander("📋 Sayisal Degerler"):
            st.table(
                {
                    "Parametre": list(etkiler.keys()),
                    "Etki (Δ Skor)": [f"{v:+.2f}" for v in etkiler.values()],
                }
            )

    # ==================================================================
    # SEKME 3: COKLU AJAN MODELI
    # ==================================================================
    with sekme_ajan:
        st.subheader("🏛️ Coklu Ajan Modeli — 435 Temsilcili Meclis Simulasyonu")
        st.markdown(
            f"""
            Tek bir "ortalama" skor yerine, her biri kendi lobi ve kutuplasma
            maruziyetine sahip **bagimsiz ajanlar** (temsilciler) hibrit
            fuzzy + sembolik boru hattindan ayri ayri gecirilir ve gercek
            bir **roll-call (oylama sayimi)** simule edilir.

            Yasayi su an **{yasayi_sunan_parti}** sunuyor. Modele **parti sadakati
            (party-line voting)** terimi eklendi: kendi partisinin yasasina
            karsi oy vermek, kutuplasma yukseldikce gittikce zorlasir; rakip
            parti uyeleri ise ayni mekanizmayla destekten uzaklasir.
            """
        )

        ajan_kol1, ajan_kol2, ajan_kol3 = st.columns(3)
        ajan_sayisi = ajan_kol1.slider("Temsilci Sayisi", 50, 435, 435, step=5)
        cumhuriyetci_orani = ajan_kol2.slider("Cumhuriyetci Orani", 0.0, 1.0, 0.51, step=0.01)
        tohum = ajan_kol3.number_input("Rastgele Tohum (Seed)", min_value=0, value=42, step=1)

        ajan_sonucu = controller.coklu_ajan_simulasyonu_calistir(
            girdi,
            ajan_sayisi=ajan_sayisi,
            cumhuriyetci_orani=cumhuriyetci_orani,
            tohum=int(tohum),
        )

        sonuc_metrik_1, sonuc_metrik_2, sonuc_metrik_3 = st.columns(3)
        sonuc_metrik_1.metric("Evet Oyu", ajan_sonucu.evet_sayisi)
        sonuc_metrik_2.metric("Hayir Oyu", ajan_sonucu.hayir_sayisi)
        sonuc_metrik_3.metric(
            "Sonuc",
            "✅ KABUL" if ajan_sonucu.kabul_edildi_mi else "❌ RET",
        )

        fig_meclis = controller.meclis_oturma_plani_grafigi(ajan_sonucu)
        st.plotly_chart(fig_meclis, use_container_width=True)

        with st.expander("🔬 Parti Bazli Kirilim"):
            for parti_adi in ["Cumhuriyetci", "Demokrat"]:
                maske = ajan_sonucu.parti == parti_adi
                if maske.sum() == 0:
                    continue
                evet = int(np.sum(ajan_sonucu.oy_evet[maske]))
                toplam = int(maske.sum())
                rol_etiketi = "sahibi (sponsor)" if parti_adi == ajan_sonucu.yasayi_sunan_parti else "muhalefet"
                st.write(
                    f"**{parti_adi}** ({rol_etiketi}): {evet}/{toplam} evet "
                    f"(%{(evet / toplam * 100):.1f}) — "
                    f"ortalama nihai skor: {ajan_sonucu.nihai_skor[maske].mean():.1f} "
                    f"| ortalama parti sadakati etkisi: {ajan_sonucu.parti_sadakati_etkisi[maske].mean():+.1f}"
                )


if __name__ == "__main__":
    arayuzu_calistir()
