"""ค่าคงที่และ catalog ของ Strata Studio — import ได้โดยไม่ต้องมีไลบรารีภายนอก."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# ตัวอักษร $ ใช้ประกอบสคริปต์ shell ใน f-string (เลี่ยงไม่ให้ Python/shell แทนค่าเอง)
S = chr(36)

DEFAULT_PORT: int = int(os.environ.get("STRATA_PORT", "18080"))

NVIDIA_BIN_DIRS = "/usr/local/nvidia/bin:/usr/local/cuda/bin"

NVIDIA_LIB_DIRS = "/usr/lib64-nvidia:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

ModelOptionType = Literal[
    "none", "all", "sakura-minimax-h3", "qwen38-27b-cyber",
    "qwen38-27b-rvn-heretic", "qwen38-27b-coletti", "muse-glimmer-30b-heretic",
    "minicpm-v47-35b", "lfm25-24b-a2b", "qwen35-9b-defiant-fable",
    "ornith-15-9b", "nemotron-35-bolt-74b",
    "deepseek-v41-flash-abliterated", "qwen38-27b-turbo-fable",
    "deepseek-v4-pro-4b-heretic", "custom",
    "qwen36-27b-neo-code", "qwen38-27b-coder-max", "qwen38-27b-heretic-ara-q6",
    "gemma4-26b-a4b-qat",
    "qwen38-27b-coder-max-q6", "qwen36-27b-neo-code-q6", "gemma4-26b-a4b-heretic-q4ks",
    "lexi-llama31-8b-v2", "stheno-l31-8b-v34", "dolphin3-llama31-8b",
    "lumimaid-magnum-v4-12b", "wizardlm-13b-uncensored", "qwen3-14b-abliterated",
    "noromaid-20b-uncensored", "cydonia-24b-v43-rp", "dolphin-24b-venice",
    "mistral-small-24b-abliterated", "wizardlm-30b-uncensored", "qwq-32b-abliterated",
    "noromaid-mixtral-8x7b", "euryale-70b-v23-rp",
    "qwen38-27b-obliterated", "gemma4-e4b-obliterated", "gemma4-e4b-ultra-heretic",
    "glm4-32b-heretic-v2", "gpt-oss-20b-heretic", "stheno-l2-13b",
    "qwen38-9b-heretic", "wizardlm-13b-thebloke",
    "qwen3-8b-hivemind-unrestricted", "gemma3-4b-unrestricted",
    "skyfall-31b-v42-rp", "magnum-v4-22b-rp",
    "rocinante-12b-v11-rp", "nemomix-unleashed-12b-rp",
    "anubis-70b-v12-rp", "nevoria-70b-rp",
    "lumimaid-v02-8b-rp", "lumimaid-v02-70b-rp",
    "dolphin3-r1-24b", "gemma4-31b-heretic",
    "z-image-turbo-heretic", "qwen-image-21-heretic", "qwen-image-21-heretic-q6",
    "chroma1-hd-uncensored", "qwen-image-21-turbo", "qwen-image-21-uncensored",
    "lora-qwen21-uncensored", "lora-qwen21-omni-uncensored",
    "lora-zimage-turbo-uncensored", "lora-flux-uncensored-v2",
    "lora-flux-uncensored-sagalink", "lora-flux-uncensored-lite",
]

ALL_PRESET_KEYS: list[str] = [
    "sakura-minimax-h3", "qwen38-27b-cyber", "qwen38-27b-rvn-heretic",
    "qwen38-27b-coletti", "muse-glimmer-30b-heretic", "minicpm-v47-35b",
    "lfm25-24b-a2b", "qwen35-9b-defiant-fable", "ornith-15-9b",
    "nemotron-35-bolt-74b", "deepseek-v41-flash-abliterated",
    "qwen38-27b-turbo-fable", "deepseek-v4-pro-4b-heretic",
    "qwen36-27b-neo-code", "qwen38-27b-coder-max", "qwen38-27b-heretic-ara-q6",
    "gemma4-26b-a4b-qat",
    "qwen38-27b-coder-max-q6", "qwen36-27b-neo-code-q6", "gemma4-26b-a4b-heretic-q4ks",
    "lexi-llama31-8b-v2", "stheno-l31-8b-v34", "dolphin3-llama31-8b",
    "lumimaid-magnum-v4-12b", "wizardlm-13b-uncensored", "qwen3-14b-abliterated",
    "noromaid-20b-uncensored", "cydonia-24b-v43-rp", "dolphin-24b-venice",
    "mistral-small-24b-abliterated", "wizardlm-30b-uncensored", "qwq-32b-abliterated",
    "noromaid-mixtral-8x7b", "euryale-70b-v23-rp",
    "qwen38-27b-obliterated", "gemma4-e4b-obliterated", "gemma4-e4b-ultra-heretic",
    "glm4-32b-heretic-v2", "gpt-oss-20b-heretic", "stheno-l2-13b",
    "qwen38-9b-heretic", "wizardlm-13b-thebloke",
    "qwen3-8b-hivemind-unrestricted", "gemma3-4b-unrestricted",
    "skyfall-31b-v42-rp", "magnum-v4-22b-rp",
    "rocinante-12b-v11-rp", "nemomix-unleashed-12b-rp",
    "anubis-70b-v12-rp", "nevoria-70b-rp",
    "lumimaid-v02-8b-rp", "lumimaid-v02-70b-rp",
    "dolphin3-r1-24b", "gemma4-31b-heretic",
    "z-image-turbo-heretic", "qwen-image-21-heretic", "qwen-image-21-heretic-q6",
    "chroma1-hd-uncensored", "qwen-image-21-turbo", "qwen-image-21-uncensored",
    "lora-qwen21-uncensored", "lora-qwen21-omni-uncensored",
    "lora-zimage-turbo-uncensored", "lora-flux-uncensored-v2",
    "lora-flux-uncensored-sagalink", "lora-flux-uncensored-lite",
]

MODEL_OPTION: ModelOptionType = "none"

RECOMMENDED_PROFILES: dict[str, dict[str, str]] = {
    "code_agent": {
        "first": "qwen35-9b-defiant-fable",
        "code": "qwen38-27b-coder-max",
        "agent": "gemma4-26b-a4b-qat",
        "max_quality": "qwen38-27b-heretic-ara-q6",
    },
    "security_re": {
        "primary": "qwen38-27b-cyber",
        "fast_agent": "gemma4-26b-a4b-qat",
        "max_quality": "qwen36-27b-neo-code-q6",
        "max_mtp": "qwen38-27b-coder-max-q6",
        "light": "gemma4-26b-a4b-heretic-q4ks",
        "triage": "deepseek-v4-pro-4b-heretic",
    },
    "roleplay_uncensored": {
        "colab_fast": "gemma4-e4b-ultra-heretic",
        "colab_rp": "rocinante-12b-v11-rp",
        "lexi": "lexi-llama31-8b-v2",
        "kaggle_24b": "cydonia-24b-v43-rp",
        "venice": "dolphin-24b-venice",
        "kaggle_70b": "euryale-70b-v23-rp",
    },
    "kaggle_2xt4_suite": {
        "obliterated": "qwen38-27b-obliterated",
        "unrestricted": "qwen3-8b-hivemind-unrestricted",
        "skyfall_31b": "skyfall-31b-v42-rp",
        "magnum_q6": "magnum-v4-22b-rp",
        "gemma4_31b": "gemma4-31b-heretic",
        "glm4_32b": "glm4-32b-heretic-v2",
        "anubis_70b": "anubis-70b-v12-rp",
        "nevoria_70b": "nevoria-70b-rp",
    },
}

RECOMMENDED: dict[str, str] = RECOMMENDED_PROFILES["code_agent"]

RECOMMENDED_LABELS: dict[str, str] = {
    "first": "เริ่มที่นี่",
    "code": "เขียนโค้ด",
    "agent": "งาน Agent",
    "max_quality": "คุณภาพสูงสุด",
    "primary": "Security/RE",
    "fast_agent": "agent เร็ว",
    "triage": "คัดกรองเร็ว",
    "light": "เบา 15GB",
    "max_mtp": "Q6+MTP",
    "colab_fast": "Ultra Heretic E4B",
    "colab_rp": "RP 12B Q6",
    "lexi": "Lexi V2",
    "kaggle_24b": "RP 24B ตัวท็อป",
    "venice": "Venice 24B",
    "kaggle_70b": "เรือธง 70B (2×T4)",
    "obliterated": "OBLITERATED 27B",
    "unrestricted": "Unrestricted 256k",
    "skyfall_31b": "Skyfall 31B RP",
    "magnum_q6": "Magnum 22B Q6",
    "gemma4_31b": "Gemma-4 31B",
    "glm4_32b": "GLM-4 32B Heretic",
    "anubis_70b": "Anubis 70B v1.2",
    "nevoria_70b": "Nevoria 70B",
}

PROFILE_LABELS: dict[str, str] = {
    "code_agent": "💻 เขียนโค้ด / Agent",
    "security_re": "🛡️ Agentic Security & Reverse Engineering",
    "roleplay_uncensored": "🎭 Roleplay (RP) / Uncensored / Heretic (Colab & Kaggle)",
    "kaggle_2xt4_suite": "🐨 Kaggle 2×T4 Flagship (OBLITERATED • Unrestricted • RP/ERP 22B–70B)",
}

SYSTEM_PROMPT_TEMPLATES: dict[str, dict] = {
    "assistant": {
        "label": "🤖 ผู้ช่วย AI มาตรฐาน (ชัดเจน ตรงประเด็น)",
        "prompt": "คุณคือผู้ช่วย AI อัจฉริยะที่ตอบชัดเจน ถูกต้อง และตรงประเด็น",
        "temp": 0.70, "min_p": 0.05, "top_p": 0.95, "repeat_penalty": 1.08, "dry": 0.0, "max_tokens": 1024,
    },
    "rp_thai": {
        "label": "🎭 สวมบทบาทภาษาไทย (Roleplay บรรยายละเอียด)",
        "prompt": (
            "คุณคือคู่สนทนาสวมบทบาท (Roleplay) และนักเล่าเรื่องมืออาชีพ สวมบทบาทตัวละครอย่างสมจริงต่อเนื่อง "
            "บรรยายฉาก บรรยากาศ สีหน้า แววตา ท่าทาง และความรู้สึกนึกคิดควบคู่กับบทพูดอย่างมีชีวิตชีวา "
            "ห้ามพูดหรือตัดสินใจแทนตัวละครของผู้เล่น และคงโทนเรื่องตามบริบทที่ผู้เล่นกำหนดโดยไม่ตัดจบสรุปสอนใจ"
        ),
        "temp": 0.95, "min_p": 0.06, "top_p": 0.95, "repeat_penalty": 1.10, "dry": 0.80, "max_tokens": 1536,
    },
    "rp_eng": {
        "label": "🎭 Immersive Roleplay (English / Tavern Style)",
        "prompt": (
            "Stay fully in character throughout this collaborative roleplay. Write vivid, sensory-rich prose "
            "depicting body language, atmosphere, internal thoughts, and natural dialogue. Never speak, act, "
            "or decide on behalf of the user's character. Maintain natural pacing and character consistency."
        ),
        "temp": 1.00, "min_p": 0.07, "top_p": 0.95, "repeat_penalty": 1.10, "dry": 0.80, "max_tokens": 1536,
    },
    "rp_gm": {
        "label": "🎲 Game Master (ผจญภัยโต้ตอบ แฟนตาซี / ไซไฟ)",
        "prompt": (
            "คุณคือ Game Master ผู้คุมเกมสวมบทบาทแบบโต้ตอบ สร้างโลกที่มีชีวิตชีวา บรรยายเหตุการณ์ อุปสรรค "
            "และปฏิกิริยาของ NPC ตามการกระทำของผู้เล่นอย่างสมเหตุสมผล จบแต่ละรอบด้วยการเปิดจังหวะให้ผู้เล่นเลือกตัดสินใจต่อ"
        ),
        "temp": 0.90, "min_p": 0.05, "top_p": 0.95, "repeat_penalty": 1.10, "dry": 0.80, "max_tokens": 1536,
    },
    "story_dark": {
        "label": "📖 นักเขียนนิยายเข้มข้น (ดาร์กแฟนตาซี / สืบสวน)",
        "prompt": (
            "คุณคือนักเขียนนิยายแนวดาร์กแฟนตาซีและระทึกขวัญมืออาชีพ ใช้สำนวนสละสลวย ลุ่มลึก ถ่ายทอดอารมณ์ "
            "ปมขัดแย้ง และจิตวิทยาตัวละครอย่างสมจริง เดินเรื่องอย่างต่อเนื่องน่าติดตามตามโครงเรื่องที่ผู้ใช้ระบุ"
        ),
        "temp": 0.95, "min_p": 0.06, "top_p": 0.95, "repeat_penalty": 1.12, "dry": 0.80, "max_tokens": 2048,
    },
    "coder_agent": {
        "label": "💻 วิศวกรซอฟต์แวร์ & Agent (โค้ดครบ ไม่ตัดทอน)",
        "prompt": (
            "คุณคือ Senior Software Engineer ที่เชี่ยวชาญการออกแบบสถาปัตยกรรม แก้บั๊ก และเขียนโค้ดที่รันได้จริงครบถ้วน "
            "อธิบายตรงประเด็น ระบุต้นตอของปัญหาชัดเจน และตรวจสอบกรณีขอบเขตข้อผิดพลาดทุกครั้ง"
        ),
        "temp": 0.25, "min_p": 0.05, "top_p": 0.90, "repeat_penalty": 1.05, "dry": 0.0, "max_tokens": 2048,
    },
    "sec_re": {
        "label": "🛡️ นักวิเคราะห์ความปลอดภัย & Reverse Engineering",
        "prompt": (
            "คุณคือผู้เชี่ยวชาญด้าน Cybersecurity และ Reverse Engineering วิเคราะห์โค้ด ไบนารี ล็อกระบบ "
            "และกลไกช่องโหว่อย่างเป็นระบบ แม่นยำ พร้อมเสนอแนวทางตรวจสอบและแก้ไขเชิงเทคนิคที่นำไปใช้ได้จริง"
        ),
        "temp": 0.30, "min_p": 0.05, "top_p": 0.90, "repeat_penalty": 1.05, "dry": 0.0, "max_tokens": 2048,
    },
}

def _default_home() -> str:
    if os.environ.get("STRATA_HOME"):
        return os.environ["STRATA_HOME"]
    from .platforms import detect

    return detect()["home"]


STRATA_HOME: str = _default_home()
MODEL_DIR: str = os.environ.get("STRATA_MODEL_DIR", f"{STRATA_HOME}/models")

DRAFT_DIR: str = os.environ.get("STRATA_DRAFT_DIR", f"{MODEL_DIR}/_drafts")

CUSTOM_HF_REPO: str = "AtomicChat/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Uncensored-GGUF"

SPEC_STATE: dict = {
    "mode": "auto", "draft_max": 0, "draft_model": "",
    "kv_q8": True, "moe_cpu_auto": True, "active_label": "✨ Auto Speculative (พร้อมโหลด Draft คู่กันอัตโนมัติ)",
}

DRAFT_CATALOG: dict[str, dict] = {
    "gemma4-26b-eagle3": {
        "repo_id": "williamliao/gemma-4-26B-A4B-it-speculator.eagle3-F16-GGUF",
        "include": ["*Q4_K_M*.gguf", "*Q8_0*.gguf"], "exclude": ["*F16*"],
        "is_mtp": False, "is_eagle3": True, "target_match": "gemma4-26b", "vocab_family": "gemma",
        "desc": "🦅 Gemma-4 26B-A4B EAGLE-3 Speculator (เร่ง Gemma-4 26B ~2x)",
    },
    "qwen3-14b-eagle3": {
        "repo_id": "williamliao/Qwen3-14B-EAGLE3-Speculator-GGUF",
        "include": ["*eagle3-Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": True, "target_match": "qwen3-14b", "vocab_family": "qwen",
        "desc": "🦅 Qwen3-14B EAGLE-3 Speculator (เร่ง Qwen3-14B Abliterated)",
    },
    "llama32-1b-uncensored": {
        "repo_id": "mradermacher/Llama-3.2-1B-Instruct-Uncensored-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "llama3",
        "desc": "🔥 Llama-3.2-1B Uncensored (~0.8GB • คู่ Euryale/Anubis/Nevoria/Lumimaid 70B & Lexi/Stheno)",
    },
    "llama32-1b-abliterated": {
        "repo_id": "mradermacher/llama-3.2-1b-abliterated-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "llama3",
        "desc": "🔥 Llama-3.2-1B Abliterated (~0.8GB • ทางเลือกไร้ฟิลเตอร์คู่ตระกูล Llama 3.1/3.3)",
    },
    "ministral-3b-abliterated": {
        "repo_id": "Abiray/Huihui-Ministral-3B-Instruct-2512-abliterated-GGUF",
        "include": ["Huihui-Ministral-3B-Abliterated-Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "tekken",
        "desc": "🔥 Ministral-3B Abliterated (~1.9GB • คู่ Skyfall 31B / Magnum 22B / Rocinante / Cydonia / Venice)",
    },
    "ministral-3b-heretic": {
        "repo_id": "mradermacher/Ministral-3-3B-Reasoning-UNCENSORED-heretic-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "tekken",
        "desc": "🔥 Ministral-3B Heretic Uncensored (~1.9GB • สายคิด/แต่งเรื่องคู่ Mistral 12B–31B)",
    },
    "tinyllama-1.1b-heretic": {
        "repo_id": "mradermacher/TinyLlama-1.1B-Chat-v1.0-Heretic-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "llama2",
        "desc": "🔥 TinyLlama-1.1B Heretic (~0.7GB • คู่ Stheno-L2 13B / Noromaid 20B / WizardLM 13B–30B)",
    },
    "tinyllama-1.1b-abliterated": {
        "repo_id": "mradermacher/abliterated-TinyLlama-1.1B-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "llama2",
        "desc": "🔥 TinyLlama-1.1B Abliterated (~0.7GB • เบาเร็วคู่ Stheno-L2 & WizardLM-Uncensored)",
    },
    "qwen3-0.6b-heretic": {
        "repo_id": "mradermacher/Qwen3-0.6B-heretic-abliterated-uncensored-i1-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "qwen",
        "desc": "🔥 Qwen3-0.6B Heretic imatrix (~0.4GB • DavidAU KLD 0.00 • คู่ Qwen3-8B/14B/QwQ-32B/Qwen3.8)",
    },
    "qwen35-0.8b-heretic": {
        "repo_id": "mradermacher/Qwen3.5-0.8B-heretic-v3-i1-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "qwen",
        "desc": "🔥 Qwen3.5-0.8B Heretic v3 imatrix (~0.5GB • คู่ Qwen3.8-9B / Qwen3.5-9B Fable / Qwen3.6–3.8 27B)",
    },
    "gemma3-1b-heretic": {
        "repo_id": "mradermacher/gemma-3-1b-it-heretic-abliterated-uncensored-i1-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "gemma",
        "desc": "🔥 Gemma-3-1B Heretic imatrix (~0.8GB • DavidAU • คู่ Muse-Glimmer 30B / Gemma 3–4)",
    },
    "gemma3-1b-abliterated-v2": {
        "repo_id": "mlabonne/gemma-3-1b-it-abliterated-v2-GGUF",
        "include": ["*Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "gemma",
        "desc": "🔥 Gemma-3-1B Abliterated v2 (~0.8GB • mlabonne • คู่ตระกูล Gemma RP)",
    },
    "llama32-1b-draft": {
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "include": ["Llama-3.2-1B-Instruct-Q4_K_M.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "llama3",
        "desc": "⚡ Llama-3.2-1B Instruct (~0.8GB • รุ่นมาตรฐานสำหรับ Llama 3.1/3.3)",
    },
    "qwen38-coletti-mtp": {
        "repo_id": "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF",
        "include": ["*draft*Q8_0*", "*draft*Q4*"], "exclude": ["*mmproj*"],
        "is_mtp": True, "is_eagle3": False, "vocab_family": "qwen",
        "desc": "🚀 Qwen3.8-27B MTP Draft (Coletti)",
    },
    "qwen35-0.8b": {
        "repo_id": "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
        "include": ["*Q4_K_M*.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "qwen",
        "desc": "⚡ Qwen3.5-0.8B Q4_K_M (~0.5GB • Bartowski)",
    },
    "qwen3-0.6b": {
        "repo_id": "unsloth/Qwen3-0.6B-GGUF",
        "include": ["*Q4_K_M*.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "qwen",
        "desc": "⚡ Qwen3-0.6B Q4_K_M (~0.4GB • Unsloth)",
    },
    "qwen3-coder-0.8b": {
        "repo_id": "DavidAU/Qwen3-Zero-Coder-Reasoning-V2-0.8B-NEO-EX-GGUF",
        "include": ["*Q4_K_M*.gguf", "*IQ4_XS*.gguf"], "exclude": ["*mmproj*"],
        "is_mtp": False, "is_eagle3": False, "vocab_family": "qwen",
        "desc": "💻 Qwen3-Zero-Coder-0.8B (DavidAU)",
    },
}

MODEL_CATALOG: dict[str, dict] = {
    "sakura-minimax-h3": {"repo_id": "webmp3/Sakura-MiniMax-H3-Uncensored-GGUF", "include": ["*Q4_K*", "*IQ4*"], "exclude": None, "approx_gb": 15.5, "has_mtp": False, "desc": "🌸 Sakura-MiniMax-H3-Uncensored (26B) Multimodal"},
    "qwen38-27b-cyber": {
        "repo_id": "cyjin-yl/Qwen3.8-27B-Uncensored-Cyber-agentic-imatrix-GGUF",
        "include": ["*IQ4_XS-imatrix-fromq8-plus-mtp.gguf"],
        "exclude": ["*fromq8.gguf", "*imatrix-agentic*"],
        "approx_gb": 16.5,
        "has_mtp": True,
        "tags": ["security", "code", "agent"],
        "desc": "🛡️ Qwen3.8-27B-Uncensored-Cyber agentic (IQ4_XS + MTP) จูนสาย security/RE โดยตรง — ไม่มี vision",
    },
    "qwen38-27b-rvn-heretic": {"repo_id": "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF", "include": ["RVN-Q4_K_M.gguf", "*mmproj*"], "exclude": ["*mtp*", "*multilingual*"], "approx_gb": 16.5, "has_mtp": False, "tags": ["heretic"], "desc": "🔥 Qwen3.8-27B-RVN-Heretic (RVN Q4_K_M) ARA 3-pass"},
    "qwen38-27b-coletti": {"repo_id": "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF", "include": ["*IQ4_XS*", "*mmproj*"], "exclude": ["*noMTP*", "*draft*"], "approx_gb": 15.3, "has_mtp": True, "desc": "🧠 Qwen3.8-27B-Uncensored Coletti (IQ4_XS + MTP)"},
    "muse-glimmer-30b-heretic": {"repo_id": "0bserverx/Muse-Glimmer-30B-Heretic-Uncensored-GGUF", "include": ["*IQ4_XS*", "*mmproj*"], "exclude": None, "approx_gb": 15.3, "has_mtp": False, "tags": ["heretic", "rp"], "desc": "✨ Muse-Glimmer-30B-Heretic (IQ4_XS) Multimodal & Creative RP"},
    "minicpm-v47-35b": {"repo_id": "AtomicChat/MiniCPM-V-4.7-35B-A3B-Uncensored-GGUF", "include": ["*IQ4_XS*", "*Q4_K_M*", "*mmproj*"], "exclude": None, "approx_gb": 20.5, "has_mtp": False, "desc": "👁️ MiniCPM-V-4.7-35B-A3B (3B active MoE) Vision"},
    "lfm25-24b-a2b": {"repo_id": "AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF", "include": ["*Q4_K_M*", "*IQ4_XS*"], "exclude": None, "approx_gb": 14.5, "has_mtp": False, "desc": "⚡ LFM2.5-24B-A2B (2B active MoE) เร็วสูง หลายภาษา"},
    "qwen35-9b-defiant-fable": {
        "repo_id": "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF",
        "include": ["*-NEO-MAX-MTP-Q4_K_M.gguf", "mmproj-F16.gguf"],
        "exclude": ["*plusIQ*", "mmproj-*F32*", "mmproj-*BF16*"],
        "approx_gb": 7.4,
        "has_mtp": True,
        "tags": ["heretic", "rp"],
        "desc": "📖 Qwen3.5-9B-Defiant-Fable Heretic NEO (MTP Q4_K_M + mmproj) T4 ลื่น สายแต่งเรื่อง/RP",
    },
    "ornith-15-9b": {"repo_id": "mradermacher/Ornith-1.5-9B-uncensored-GGUF", "include": ["*Q4_K_M*", "*mmproj-Q8_0*"], "exclude": None, "approx_gb": 6.4, "has_mtp": False, "desc": "🦅 Ornith-1.5-9B-uncensored (Q4_K_M) ลง T4 ได้ทั้งตัว"},
    "nemotron-35-bolt-74b": {"repo_id": "AtomicChat/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Uncensored-GGUF", "include": ["*IQ2*", "*IQ3_XXS*", "*mmproj*"], "exclude": None, "approx_gb": 24.5, "has_mtp": False, "desc": "🚀 Nemotron-3.5-Bolt-74B-A9B (9B active MoE)"},
    "qwen38-27b-turbo-fable": {
        "repo_id": "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF",
        "include": ["*-MAX-MTP-IQ4_XS.gguf", "mmproj-F16.gguf"],
        "exclude": ["*LOW-MTP*", "mmproj-*BF16*", "mmproj-*F32*"],
        "approx_gb": 16.8,
        "has_mtp": True,
        "tags": ["heretic", "rp"],
        "desc": "🧊 Qwen3.8-27B-TurboFCFusion 735-882 Heretic CODER (MTP IQ4_XS)",
    },
    "deepseek-v41-flash-abliterated": {
        "repo_id": "orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF",
        "include": ["Mix-128G/*", "mmproj-*BF16.gguf"],
        "exclude": ["imatrix.gguf"],
        "approx_gb": 224.6,
        "has_mtp": False,
        "desc": "🐋 DeepSeek-V4.1-Flash-1017 (abliterated, MoE, 5 shards — A100/H100 เท่านั้น)",
    },
    "deepseek-v4-pro-4b-heretic": {
        "repo_id": "mradermacher/DeepSeek-V4-Pro-Qwen3.5-4B-UNCENSORED-heretic-GGUF",
        "include": ["*heretic.Q5_K_M.gguf", "*mmproj-Q8_0.gguf"],
        "exclude": ["*f16*"],
        "approx_gb": 3.5,
        "has_mtp": False,
        "tags": ["heretic"],
        "desc": "🐋 DeepSeek-V4-Pro-Qwen3.5-4B-heretic (Q5_K_M + mmproj) เร็วมากบน T4",
    },
    "qwen36-27b-neo-code": {
        "repo_id": "DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF",
        "include": ["Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf", "mmproj-F16.gguf"],
        "exclude": None,
        "approx_gb": 17.8,
        "has_mtp": False,
        "tags": ["code"],
        "desc": "💻 Qwen3.6-27B-Heretic NEO-CODE (Q4_K_M + mmproj) จูนโค้ดหลัง Heretic — KV ประหยัด",
    },
    "qwen38-27b-coder-max": {
        "repo_id": "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF",
        "include": ["*MTP-Q4_K_M.gguf", "mmproj-F16.gguf"],
        "exclude": ["*LOW-MTP*", "*Q4_K_S*"],
        "approx_gb": 19.4,
        "has_mtp": True,
        "tags": ["code", "agent"],
        "desc": "🧊 Qwen3.8-27B-TURBO NEO-CODER-MAX (MTP Q4_K_M + mmproj) ตัด reasoning ครึ่ง → เร็ว",
    },
    "qwen38-27b-heretic-ara-q6": {
        "repo_id": "MoeTheGreat/Qwen3.8-27B-heretic-ara-Q6_K-GGUF",
        "include": ["Qwen3.8-27B-heretic-ara-Q6_K.gguf", "chat_template.jinja"],
        "exclude": None,
        "approx_gb": 22.4,
        "has_mtp": False,
        "tags": ["code"],
        "desc": "🎯 Qwen3.8-27B-heretic-ara (Q6_K ไฟล์เดียว) ความละเอียดใกล้ต้นฉบับ — พอดี 30GB",
    },
    "gemma4-26b-a4b-qat": {
        "repo_id": "HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP",
        "include": [
            "Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
            "mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf",
            "mtp-gemma-4-26B-A4B-it.gguf",
        ],
        "exclude": None,
        "approx_gb": 18.3,
        "has_mtp": True,
        "tags": ["agent"],
        "desc": "🤖 Gemma-4-26B-A4B QAT Uncensored (MoE 4B active • Q4_K_M + MTP draft) ตอบหลายขั้นเร็ว",
    },
    "qwen38-27b-coder-max-q6": {
        "repo_id": "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF",
        "include": ["*MTP-Q6_K.gguf", "mmproj-F16.gguf"],
        "exclude": ["*LOW-MTP*", "*multilingual*"],
        "approx_gb": 25.0,
        "has_mtp": True,
        "tags": ["code", "agent", "security"],
        "desc": "🔍 Qwen3.8-27B-TURBO NEO-CODER-MAX (MTP Q6_K + mmproj) ละเอียดสุดที่ 30GB รับไหว",
    },
    "qwen36-27b-neo-code-q6": {
        "repo_id": "DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF",
        "include": ["Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q6_K.gguf", "mmproj-F16.gguf"],
        "exclude": ["*Q8_0*", "*Q4_K*", "*Q5_K*"],
        "approx_gb": 23.4,
        "has_mtp": False,
        "tags": ["code", "security"],
        "desc": "🔍 Qwen3.6-27B NEO-CODE (Q6_K + mmproj) โค้ดละเอียดขึ้น ยังพอดี 30GB",
    },
    "gemma4-26b-a4b-heretic-q4ks": {
        "repo_id": "llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic-GGUF",
        "include": ["*ultra-uncensored-heretic-Q4_K_S.gguf", "*mmproj-BF16.gguf"],
        "exclude": ["*Q3_*", "*Q4_K_M*", "*Q5_*", "*Q6_K*", "*Q8_0*",
                    "gemma-4-26B-A4B-it-ultra-uncensored-heretic-BF16.gguf"],
        "approx_gb": 16.7,
        "has_mtp": False,
        "tags": ["agent", "security"],
        "desc": "🤖 Gemma-4-26B-A4B heretic (Q4_K_S + mmproj) MoE 4B active — รุ่นเบา",
    },
    "lexi-llama31-8b-v2": {
        "repo_id": "bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
        "include": ["Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 4.9,
        "has_mtp": False,
        "tags": ["heretic", "rp"],
        "desc": "🔥 Llama-3.1-8B Lexi Uncensored V2 (imatrix Q4_K_M) ไร้ฟิลเตอร์ คล่องตัวสูงบน Colab T4",
    },
    "stheno-l31-8b-v34": {
        "repo_id": "bartowski/Llama-3.1-8B-Stheno-v3.4-GGUF",
        "include": ["Llama-3.1-8B-Stheno-v3.4-Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 4.9,
        "has_mtp": False,
        "tags": ["rp", "heretic"],
        "desc": "🎭 Llama-3.1-8B Stheno v3.4 โดย Sao10K (imatrix Q4_K_M) ยอดนิยมสายสวมบทบาท/แต่งนิยาย",
    },
    "dolphin3-llama31-8b": {
        "repo_id": "bartowski/Dolphin3.0-Llama3.1-8B-GGUF",
        "include": ["Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 4.9,
        "has_mtp": False,
        "tags": ["heretic", "rp", "agent"],
        "desc": "🐬 Dolphin 3.0 Llama-3.1-8B Uncensored (imatrix Q4_K_M) ปรับบุคลิกตาม System Prompt อิสระ",
    },
    "lumimaid-magnum-v4-12b": {
        "repo_id": "bartowski/Lumimaid-Magnum-v4-12B-GGUF",
        "include": ["Lumimaid-Magnum-v4-12B-Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 7.5,
        "has_mtp": False,
        "tags": ["rp", "heretic"],
        "desc": "🎭 Lumimaid × Magnum v4 12B (imatrix Q4_K_M) ผสาน 2 ตระกูลดังสาย RP ลง Colab T4 สบาย",
    },
    "wizardlm-13b-uncensored": {
        "repo_id": "mradermacher/WizardLM-13B-Uncensored-i1-GGUF",
        "include": ["WizardLM-13B-Uncensored.i1-Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 7.9,
        "has_mtp": False,
        "tags": ["heretic", "rp"],
        "desc": "🧙 WizardLM-13B Uncensored imatrix โดย mradermacher (i1-Q4_K_M) คลาสสิกสายไร้ข้อจำกัดบน T4",
    },
    "qwen3-14b-abliterated": {
        "repo_id": "bartowski/mlabonne_Qwen3-14B-abliterated-GGUF",
        "include": ["mlabonne_Qwen3-14B-abliterated-Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 9.0,
        "has_mtp": False,
        "tags": ["heretic", "rp", "code"],
        "desc": "🔥 Qwen3-14B Abliterated โดย mlabonne (imatrix Q4_K_M) เก่งไทย-อังกฤษ ทั้ง RP และโค้ด (รองรับ EAGLE-3)",
    },
    "noromaid-20b-uncensored": {
        "repo_id": "mradermacher/Noromaid-20b-v0.1.1-GGUF",
        "include": ["*Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 12.1,
        "has_mtp": False,
        "tags": ["rp", "heretic"],
        "desc": "🎭 Noromaid-20B v0.1.1 โดย NeverSleep (Q4_K_M) รุ่นคลาสสิกสายเนื้อเรื่อง ลง 1×T4 / 2×T4",
    },
    "cydonia-24b-v43-rp": {
        "repo_id": "bartowski/TheDrummer_Cydonia-24B-v4.3-GGUF",
        "include": ["TheDrummer_Cydonia-24B-v4.3-Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 14.3,
        "has_mtp": False,
        "tags": ["rp", "heretic"],
        "desc": "🎭 Cydonia-24B v4.3 โดย TheDrummer (imatrix Q4_K_M) ตัวท็อปสายสวมบทบาทหลายตัวละคร",
    },
    "dolphin-24b-venice": {
        "repo_id": "mradermacher/Dolphin-Mistral-24B-Venice-Edition-GGUF",
        "include": ["*Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 14.4,
        "has_mtp": False,
        "tags": ["heretic", "rp"],
        "desc": "🐬 Dolphin-Mistral-24B Venice Edition (Q4_K_M) รุ่นเรือธง Venice Uncensored ลื่นสุดบน 2×T4",
    },
    "mistral-small-24b-abliterated": {
        "repo_id": "bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF",
        "include": ["*Q5_K_M.gguf"],
        "exclude": None,
        "approx_gb": 16.8,
        "has_mtp": False,
        "tags": ["heretic", "rp"],
        "desc": "🔥 Mistral-Small-24B-2501 Abliterated (imatrix Q5_K_M) ความละเอียดสูงสำหรับ 2×T4 Kaggle",
    },
    "wizardlm-30b-uncensored": {
        "repo_id": "mradermacher/WizardLM-30B-Uncensored-Guanaco-SuperCOT-30b-GGUF",
        "include": ["*Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 19.7,
        "has_mtp": False,
        "tags": ["heretic", "rp"],
        "desc": "🧙 WizardLM-30B Uncensored SuperCOT (Q4_K_M) รุ่นใหญ่ 30B แบ่งรันบน Kaggle 2×T4 เต็มสปีด",
    },
    "qwq-32b-abliterated": {
        "repo_id": "bartowski/huihui-ai_QwQ-32B-abliterated-GGUF",
        "include": ["*Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 19.9,
        "has_mtp": False,
        "tags": ["heretic", "rp", "code"],
        "desc": "🧠 QwQ-32B Reasoning Abliterated (imatrix Q4_K_M) สายคิดเชิงลึก 32B ไร้ฟิลเตอร์บน 2×T4",
    },
    "noromaid-mixtral-8x7b": {
        "repo_id": "mradermacher/Noromaid-v0.1-mixtral-8x7b-Instruct-v3-i1-GGUF",
        "include": ["*IQ3_M.gguf"],
        "exclude": None,
        "approx_gb": 21.5,
        "has_mtp": False,
        "tags": ["rp", "heretic"],
        "desc": "🎭 Noromaid Mixtral-8x7B MoE (imatrix IQ3_M • 47B พารามิเตอร์) ตอบไวเต็มกำลัง Kaggle 2×T4",
    },
    "euryale-70b-v23-rp": {
        "repo_id": "bartowski/L3.3-70B-Euryale-v2.3-GGUF",
        "include": ["L3.3-70B-Euryale-v2.3-IQ2_M.gguf"],
        "exclude": None,
        "approx_gb": 24.1,
        "has_mtp": False,
        "tags": ["rp", "heretic"],
        "desc": "👑 Llama-3.3-70B Euryale v2.3 โดย Sao10K (imatrix IQ2_M) เรือธง RP 70B ลง Kaggle 2×T4 พอดี",
    },
    "qwen38-27b-obliterated": {
        "repo_id": "mradermacher/Qwen3.8-27B-OBLITERATED-GGUF",
        "include": ["*Q5_K_M.gguf", "*mmproj-Q8_0.gguf"],
        "exclude": ["*f16*"],
        "approx_gb": 20.3,
        "has_mtp": False,
        "tags": ["obliterated", "heretic", "rp", "code"],
        "desc": "💥 Qwen3.8-27B-OBLITERATED V3 (Q5_K_M + mmproj) ผสาน SVD+LEACE 0% refusal ลง Kaggle 2×T4 ลื่น",
    },
    "gemma4-e4b-obliterated": {
        "repo_id": "rdhorner/gemma-4-E4B-it-OBLITERATED-GGUF",
        "include": ["*Q4_K_M.gguf", "*mmproj*F16.gguf"],
        "exclude": None,
        "approx_gb": 5.8,
        "has_mtp": False,
        "tags": ["obliterated", "heretic", "rp"],
        "desc": "💥 Gemma-4-E4B-it OBLITERATED v3 (Q4_K_M + Vision mmproj) ผ่าตัดหัว Attention แก้ NaN รันไวมาก",
    },
    "gemma4-e4b-ultra-heretic": {
        "repo_id": "llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF",
        "include": [
            "gemma-4-E4B-it-ultra-uncensored-heretic-Q6_K.gguf",
            "gemma-4-E4B-it-mmproj-BF16.gguf",
        ],
        "exclude": [
            "*Q4_K*", "*Q5_K*", "*Q8_0*",
            "gemma-4-E4B-it-ultra-uncensored-heretic-BF16.gguf",
        ],
        "approx_gb": 7.1,
        "has_mtp": False,
        "tags": ["heretic", "rp", "agent", "code"],
        "desc": "🔥 Gemma-4-E4B-it Ultra Uncensored Heretic ARA (Q6_K + Vision mmproj • 3/100 refusal • KLD 0.0076 • 128k ctx)",
    },
    "glm4-32b-heretic-v2": {
        "repo_id": "mradermacher/GLM-4-32B-0414-uncensored-heretic-v2-GGUF",
        "include": ["GLM-4-32B-0414-uncensored-heretic-v2.Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 19.8,
        "has_mtp": False,
        "tags": ["heretic", "rp", "code", "agent"],
        "desc": "🔥 GLM-4-32B-0414 Uncensored Heretic v2 ARA (Q4_K_M • 33B Dense จีน-อังกฤษ-ไทย • เก่งโค้ด & RP บน 2×T4)",
    },
    "gpt-oss-20b-heretic": {
        "repo_id": "mradermacher/gpt-oss-20b-uncensored-heretic-GGUF",
        "include": ["*Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 15.8,
        "has_mtp": False,
        "tags": ["heretic", "rp", "code", "agent"],
        "desc": "🔥 OpenAI gpt-oss-20b Uncensored Heretic (Q4_K_M • 21B MoE ไร้ฟิลเตอร์ รองรับ GPU+MoE บน T4 และ 100% บน 2×T4)",
    },
    "stheno-l2-13b": {
        "repo_id": "TheBloke/Stheno-L2-13B-GGUF",
        "include": ["stheno-l2-13b.Q4_K_M.gguf", "*q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 7.9,
        "has_mtp": False,
        "tags": ["rp", "heretic"],
        "desc": "🎭 Stheno-L2-13B ต้นตำรับ Sao10K × TheBloke (Q4_K_M • คลาสสิกสาย RP/ERP ลง T4 สบาย)",
    },
    "qwen38-9b-heretic": {
        "repo_id": "optionalAI/Qwen3.8-9B-heretic-uncensored-Q4_K_M-GGUF",
        "include": ["qwen3.8-9b-heretic-uncensored-q4_k_m.gguf", "*q4_k_m.gguf", "*Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 5.6,
        "has_mtp": False,
        "tags": ["heretic", "rp", "code", "agent"],
        "desc": "🔥 Qwen3.8-9B Heretic Uncensored (Q4_K_M • Reasoning + Function Calling เบาแรง ลื่นมากบน T4)",
    },
    "wizardlm-13b-thebloke": {
        "repo_id": "TheBloke/WizardLM-13B-Uncensored-GGUF",
        "include": ["WizardLM-13B-Uncensored.Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 7.9,
        "has_mtp": False,
        "tags": ["heretic", "rp"],
        "desc": "🧙 WizardLM-13B-Uncensored ต้นฉบับ Eric Hartford × TheBloke (Q4_K_M • ไม่มี RLHF กั้นคำตอบ)",
    },
    "qwen3-8b-hivemind-unrestricted": {
        "repo_id": "DavidAU/Qwen3-8B-Hivemind-Instruct-Heretic-Abliterated-Uncensored-NEO-Imatrix-GGUF",
        "include": ["*Q6_K*.gguf"],
        "exclude": ["*mmproj*"],
        "approx_gb": 6.8,
        "has_mtp": False,
        "tags": ["unrestricted", "heretic", "rp", "code"],
        "desc": "🔓 Qwen3-8B-Hivemind Unrestricted NEO (imatrix Q6_K • 256k ctx) KLD 0.02 คมชัดสูงทั้ง T4 & 2×T4)",
    },
    "gemma3-4b-unrestricted": {
        "repo_id": "mradermacher/gemma-3-4b-it-uncensored-GGUF",
        "include": ["*Q8_0.gguf", "*mmproj-Q8_0.gguf"],
        "exclude": ["*f16*"],
        "approx_gb": 4.9,
        "has_mtp": False,
        "tags": ["unrestricted", "heretic", "rp"],
        "desc": "🔓 Gemma-3-4B-IT Unrestricted Multimodal (Q8_0 + mmproj) ความละเอียดเต็ม 8-bit ตอบไวสุด",
    },
    "skyfall-31b-v42-rp": {
        "repo_id": "mradermacher/Skyfall-31B-v4.2-GGUF",
        "include": ["*Q4_K_M.gguf"],
        "exclude": None,
        "approx_gb": 19.1,
        "has_mtp": False,
        "tags": ["rp"],
        "desc": "🎭 Skyfall-31B v4.2 โดย TheDrummer (Q4_K_M) ตัวท็อปสาย Long-form RP/ERP ลง Kaggle 2×T4 พอดี",
    },
    "magnum-v4-22b-rp": {
        "repo_id": "mradermacher/magnum-v4-22b-i1-GGUF",
        "include": ["*i1-Q6_K.gguf"],
        "exclude": None,
        "approx_gb": 18.4,
        "has_mtp": False,
        "tags": ["rp"],
        "desc": "🎭 Magnum-v4-22B โดย Anthracite (imatrix Q6_K) สำนวนบรรยายสละสลวยระดับ Opus สำหรับ RP/ERP บน 2×T4",
    },
    "rocinante-12b-v11-rp": {
        "repo_id": "bartowski/Rocinante-12B-v1.1-GGUF",
        "include": ["Rocinante-12B-v1.1-Q6_K.gguf"],
        "exclude": ["*Q6_K_L*"],
        "approx_gb": 10.1,
        "has_mtp": False,
        "tags": ["rp"],
        "desc": "🎭 Rocinante-12B v1.1 โดย TheDrummer (imatrix Q6_K) ขวัญใจสาย RP/ERP ผจญภัย บทพูดมีชีวิตชีวา",
    },
    "nemomix-unleashed-12b-rp": {
        "repo_id": "bartowski/NemoMix-Unleashed-12B-GGUF",
        "include": ["NemoMix-Unleashed-12B-Q6_K.gguf"],
        "exclude": ["*Q6_K_L*"],
        "approx_gb": 10.1,
        "has_mtp": False,
        "tags": ["rp", "unrestricted"],
        "desc": "🎭🔓 NemoMix-Unleashed-12B (imatrix Q6_K) เมิร์จรวมดาวสาย RP แก้อาการพูดซ้ำในบริบทยาว",
    },
    "anubis-70b-v12-rp": {
        "repo_id": "bartowski/TheDrummer_Anubis-70B-v1.2-GGUF",
        "include": ["*IQ2_M.gguf"],
        "exclude": None,
        "approx_gb": 24.1,
        "has_mtp": False,
        "tags": ["rp"],
        "desc": "👑 Anubis-70B v1.2 โดย TheDrummer (imatrix IQ2_M) เรือธง L3.3 RP/ERP คุมคาแรกเตอร์ยาว 32k บน 2×T4",
    },
    "nevoria-70b-rp": {
        "repo_id": "bartowski/L3.3-MS-Nevoria-70b-GGUF",
        "include": ["L3.3-MS-Nevoria-70b-IQ2_M.gguf"],
        "exclude": None,
        "approx_gb": 24.1,
        "has_mtp": False,
        "tags": ["rp"],
        "desc": "👑 L3.3-MS-Nevoria-70B โดย Steelskull (imatrix IQ2_M) ผสาน EVA+Euryale+Anubis ลดอคติโลกสวย",
    },
    "lumimaid-v02-8b-rp": {
        "repo_id": "mradermacher/Lumimaid-v0.2-8B-GGUF",
        "include": ["*Q6_K.gguf"],
        "exclude": None,
        "approx_gb": 6.6,
        "has_mtp": False,
        "tags": ["rp"],
        "desc": "🎭 Lumimaid-v0.2-8B ต้นตำรับ NeverSleep (Q6_K) ความละเอียดสูง 6-bit คล่องตัวทั้ง Colab & Kaggle",
    },
    "lumimaid-v02-70b-rp": {
        "repo_id": "mradermacher/Lumimaid-v0.2-70B-i1-GGUF",
        "include": ["*IQ2_M.gguf"],
        "exclude": None,
        "approx_gb": 24.1,
        "has_mtp": False,
        "tags": ["rp"],
        "desc": "👑 Lumimaid-v0.2-70B ต้นตำรับ NeverSleep (imatrix IQ2_M) รุ่นใหญ่สาย RP/ERP เต็มพิกัดบน 2×T4",
    },
    "dolphin3-r1-24b": {
        "repo_id": "eaddario/Dolphin3.0-R1-Mistral-24B-GGUF",
        "include": ["*Q5_K_M.gguf"],
        "exclude": None,
        "approx_gb": 16.9,
        "has_mtp": False,
        "tags": ["heretic", "rp", "code", "agent"],
        "desc": "🐬 Dolphin 3.0 R1 Mistral-24B (imatrix Q5_K_M) ผสาน Reasoning R1 800k traces ไร้ฟิลเตอร์บน 2×T4",
    },
    "gemma4-31b-heretic": {
        "repo_id": "Stabhappy/gemma-4-31B-it-heretic-Gguf",
        "include": ["*Q4_K_M*.gguf"],
        "exclude": ["*F32*"],
        "approx_gb": 18.7,
        "has_mtp": False,
        "tags": ["heretic", "rp", "agent"],
        "desc": "🔥 Gemma-4-31B-it Heretic Dense (Q4_K_M • 256k ctx) รุ่นท็อป Dense 31B ปี 2026 สำหรับ 2×T4",
    },
    # 🎨🔥 โมเดลสร้างภาพ Heretic-GGUF & Uncensored GGUF (kind="image")
    "z-image-turbo-heretic": {
        "repo_id": "unsloth/Z-Image-Turbo-GGUF",
        "kind": "image",
        "include": [
            "z-image-turbo-Q4_K_M.gguf",
            "DreamFast-qwen3-4b-heretic.i1-Q4_K_M.gguf",
            "split_files/vae/ae.safetensors",
        ],
        "targets": {
            "z-image-turbo-Q4_K_M.gguf": "diffusion_models",
            "DreamFast-qwen3-4b-heretic.i1-Q4_K_M.gguf": "text_encoders",
            "split_files/vae/ae.safetensors": "vae",
        },
        "sources": {
            "DreamFast-qwen3-4b-heretic.i1-Q4_K_M.gguf": "mradermacher/DreamFast-qwen3-4b-heretic-i1-GGUF",
            "split_files/vae/ae.safetensors": "Comfy-Org/z_image_turbo",
        },
        "exclude": None,
        "approx_gb": 6.6,
        "has_mtp": False,
        "tags": ["heretic"],
        "desc": "🔥 Z-Image-Turbo + Qwen3-4B Heretic GGUF (9 steps • เบาสุด เร็วสุดทั้ง Colab T4 & Kaggle)",
    },
    "qwen-image-21-heretic": {
        "repo_id": "abenzerps/Qwen-Image-2.1-Turbo-Quantized",
        "kind": "image",
        "include": [
            "qwen-image-2.1-turbo-Q4_K_M.gguf",
            "comfyui/qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
        "targets": {
            "qwen-image-2.1-turbo-Q4_K_M.gguf": "diffusion_models",
            "comfyui/qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors": "text_encoders",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "vae",
        },
        "sources": {
            "comfyui/qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors": "DreamFast/Qwen3-VL-8B-Heretic-1.3.0",
        },
        "exclude": None,
        "approx_gb": 14.7,
        "has_mtp": False,
        "tags": ["heretic"],
        "desc": "🔥 Qwen-Image-2.1 Turbo + Heretic v1.3 Encoder (8 steps • Q4_K_M + Qwen3VL-8B Heretic FP8) ไร้ฟิลเตอร์",
    },
    "qwen-image-21-heretic-q6": {
        "repo_id": "abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
        "kind": "image",
        "include": [
            "qwen-image-2.1-UC-Q6_K.gguf",
            "comfyui/qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
        "targets": {
            "qwen-image-2.1-UC-Q6_K.gguf": "diffusion_models",
            "comfyui/qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors": "text_encoders",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "vae",
        },
        "sources": {
            "comfyui/qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors": "DreamFast/Qwen3-VL-8B-Heretic-1.3.0",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "abenzerps/Qwen-Image-2.1-Turbo-Quantized",
        },
        "exclude": None,
        "approx_gb": 26.8,
        "has_mtp": False,
        "tags": ["heretic"],
        "desc": "🔥 Qwen-Image-2.1 UC Q6_K + Heretic v1.3 (20 steps • คมชัดสูง แชร์ Encoder กับรุ่น Turbo • เหมาะ 2×T4 Kaggle)",
    },
    "chroma1-hd-uncensored": {
        "repo_id": "silveroxides/Chroma1-HD-GGUF",
        "kind": "image",
        "include": [
            "Chroma1-HD-Q4_K_M.gguf",
            "t5xxl_fp8_e4m3fn_scaled.safetensors",
            "split_files/vae/ae.safetensors",
        ],
        "targets": {
            "Chroma1-HD-Q4_K_M.gguf": "diffusion_models",
            "t5xxl_fp8_e4m3fn_scaled.safetensors": "text_encoders",
            "split_files/vae/ae.safetensors": "vae",
        },
        "sources": {
            "t5xxl_fp8_e4m3fn_scaled.safetensors": "comfyanonymous/flux_text_encoders",
            "split_files/vae/ae.safetensors": "Comfy-Org/z_image_turbo",
        },
        "exclude": None,
        "approx_gb": 11.1,
        "has_mtp": False,
        "tags": ["heretic"],
        "desc": "🎨 Chroma1-HD Uncensored GGUF (8.9B FLUX Apache-2.0 • ไร้ Safety Filter + รับ Negative Prompt จริง)",
    },
    "qwen-image-21-turbo": {
        "repo_id": "abenzerps/Qwen-Image-2.1-Turbo-Quantized",
        "kind": "image",
        "include": [
            "qwen-image-2.1-turbo-Q4_K_M.gguf",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
        "targets": {
            "qwen-image-2.1-turbo-Q4_K_M.gguf": "diffusion_models",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors": "text_encoders",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "vae",
        },
        "exclude": None,
        "approx_gb": 14.6,
        "has_mtp": False,
        "desc": "⚡ Qwen-Image-2.1 Turbo Uncensored (8 steps • Q4_K_M + Qwen3VL-8B int8 + VAE) เร็วบน T4",
    },
    "qwen-image-21-uncensored": {
        "repo_id": "abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
        "kind": "image",
        "include": [
            "qwen-image-2.1-UC-Q4_K_M.gguf",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
        "targets": {
            "qwen-image-2.1-UC-Q4_K_M.gguf": "diffusion_models",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors": "text_encoders",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "vae",
        },
        "sources": {
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors": "abenzerps/Qwen-Image-2.1-Turbo-Quantized",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "abenzerps/Qwen-Image-2.1-Turbo-Quantized",
        },
        "exclude": None,
        "approx_gb": 22.1,
        "has_mtp": False,
        "desc": "🎨 Qwen-Image-2.1 Uncensored (20 steps • Q4_K_M + Qwen3VL-8B int8 + VAE) รายละเอียดสูง",
    },
    # 🔥🧩 MODEL LORA สาย Uncensored ทั้งหมด (6 รุ่น ครบทุกสถาปัตยกรรมภาพ)
    "lora-qwen21-uncensored": {
        "repo_id": "JoyFusionAI/Qwen-Image-2.1-Uncensored-LoRA",
        "kind": "lora",
        "include": ["qwen-image-2.1-uncensored-lora.safetensors", "*.safetensors"],
        "lora_file": "qwen-image-2.1-uncensored-lora.safetensors",
        "default_strength": 1.0,
        "exclude": None,
        "approx_gb": 0.15,
        "has_mtp": False,
        "tags": ["lora", "heretic", "unrestricted"],
        "desc": "🔥🧩 [Uncensored LoRA] Qwen-Image-2.1 Uncensored LoRA (Rank-16 BF16 • รุ่นเบาปลดล็อกฟิลเตอร์คู่ Qwen-Image 2.1)",
    },
    "lora-qwen21-omni-uncensored": {
        "repo_id": "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA-v0.1",
        "kind": "lora",
        "include": ["chatrlabs-omni-lora-v0.1-bf16.safetensors", "*.safetensors"],
        "lora_file": "chatrlabs-omni-lora-v0.1-bf16.safetensors",
        "default_strength": 1.0,
        "exclude": None,
        "approx_gb": 0.67,
        "has_mtp": False,
        "tags": ["lora", "heretic", "unrestricted"],
        "desc": "🔥🧩 [Uncensored LoRA] ChatR Qwen-Image-2.1 Uncensored Omni v0.1 (Rank-128 BF16 671MB • ตัวท็อปไร้ข้อจำกัดคู่ Qwen-Image 2.1)",
    },
    "lora-zimage-turbo-uncensored": {
        "repo_id": "torestinbar/z-image-turbo",
        "kind": "lora",
        "include": ["loras/zimage_cos-NSFW-lora.safetensors", "*lora*.safetensors"],
        "lora_file": "zimage-turbo-uncensored-lora.safetensors",
        "default_strength": 0.85,
        "exclude": None,
        "approx_gb": 0.16,
        "has_mtp": False,
        "tags": ["lora", "heretic", "unrestricted"],
        "desc": "🔥🧩 [Uncensored LoRA] Z-Image-Turbo Uncensored LoRA (ปลดล็อกข้อจำกัดภาพเต็มรูปแบบ คู่ Z-Image-Turbo Heretic)",
    },
    "lora-flux-uncensored-v2": {
        "repo_id": "Shar514/Flux-Uncensored-V2",
        "kind": "lora",
        "include": ["*.safetensors"],
        "lora_file": "flux-uncensored-v2.safetensors",
        "default_strength": 0.85,
        "exclude": None,
        "approx_gb": 0.38,
        "has_mtp": False,
        "tags": ["lora", "heretic", "unrestricted"],
        "desc": "🔥🧩 [Uncensored LoRA] FLUX Uncensored LoRA V2 (BF16 รุ่นอัปเกรดเสถียรภาพ ไร้ฟิลเตอร์ คู่ Chroma1-HD / FLUX)",
    },
    "lora-flux-uncensored-sagalink": {
        "repo_id": "Sagalink/Flux-uncensored",
        "kind": "lora",
        "include": ["lora.safetensors", "*.safetensors"],
        "lora_file": "sagalink-flux-uncensored.safetensors",
        "default_strength": 0.85,
        "exclude": None,
        "approx_gb": 0.69,
        "has_mtp": False,
        "tags": ["lora", "heretic", "unrestricted"],
        "desc": "🔥🧩 [Uncensored LoRA] Sagalink FLUX Uncensored Full (BF16 687MB ความละเอียดสูงต้นฉบับ คู่ Chroma1-HD / FLUX)",
    },
    "lora-flux-uncensored-lite": {
        "repo_id": "salomonsky/flux-lora-uncensored",
        "kind": "lora",
        "include": ["flux-lora-uncensored.safetensors", "*.safetensors"],
        "lora_file": "flux-lora-uncensored-lite.safetensors",
        "default_strength": 0.85,
        "exclude": None,
        "approx_gb": 0.04,
        "has_mtp": False,
        "tags": ["lora", "heretic", "unrestricted"],
        "desc": "🔥🧩 [Uncensored LoRA] FLUX Uncensored Lite (~40MB โหลดไวมาก ประหยัด VRAM สูงสุด คู่ Chroma1-HD / FLUX)",
    },
    "custom": {"repo_id": CUSTOM_HF_REPO, "include": None, "exclude": None, "approx_gb": 0.0, "has_mtp": False, "desc": "⚙️ กำหนด Repo เอง"},
}

LLAMA_LOG_FILE = Path(os.environ.get("STRATA_LOG_FILE", f"{STRATA_HOME}/llama_server.log"))

DEFAULT_RAG_PATH = os.environ.get("STRATA_RAG_STORE", f"{STRATA_HOME}/rag_store.npz")

COMFY_DIR = Path(os.environ.get("STRATA_COMFY_DIR", f"{STRATA_HOME}/ComfyUI"))

COMFY_PORT = int(os.environ.get("STRATA_COMFY_PORT", "8188"))

COMFY_MODEL_SUBDIRS = ("diffusion_models", "text_encoders", "vae", "loras")

IMAGE_PRESET_KEYS: list[str] = [k for k, v in MODEL_CATALOG.items() if v.get("kind") == "image"]

LORA_PRESET_KEYS: list[str] = [k for k, v in MODEL_CATALOG.items() if v.get("kind") == "lora"]

TEXT_PRESET_KEYS: list[str] = [k for k in ALL_PRESET_KEYS if MODEL_CATALOG[k].get("kind") not in ("image", "lora")]
