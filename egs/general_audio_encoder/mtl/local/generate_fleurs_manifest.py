import logging

from lhotse import CutSet
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet

DEFAULT_LANGUAGES = [
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
]

def main():
    manifest_root = "data/fleurs_manifest"
    for language in DEFAULT_LANGUAGES:

        for split in ["train", "dev", "test"]:
            logging.info(f"Processing {language}: {split}")
            recording_file = f"{manifest_root}/{language}/fleurs-{language}_recordings_{split}.jsonl.gz"
            supervision_file = f"{manifest_root}/{language}/fleurs-{language}_supervisions_{split}.jsonl.gz"
            recordings = RecordingSet.from_file(recording_file)
            supervisions = SupervisionSet.from_file(supervision_file)
            cut_set = CutSet.from_manifests(
                recordings=recordings,
                supervisions=supervisions,
            )
            output_manifest = f"data/fleurs_manifest/{language}/fleurs_cuts_{language}_{split}.jsonl.gz"
            cut_set.to_jsonl(output_manifest)
            
        
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()