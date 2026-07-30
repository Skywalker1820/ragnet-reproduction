#!/usr/bin/env bash
set -euo pipefail

# Persistent locations on TIGRIS. They can be overridden for another machine.
RAGNET_DATA_ROOT="${RAGNET_DATA_ROOT:-/shared/rc/mairl/datasets/ragnet/processed/data}"
RAGNET_RAW_ROOT="${RAGNET_RAW_ROOT:-/shared/rc/mairl/datasets/ragnet/raw/lisa}"
RAGNET_MODEL_ROOT="${RAGNET_MODEL_ROOT:-/shared/rc/mairl/results/ragnet-reproduction/models}"

LISA_ROOT="${RAGNET_DATA_ROOT}/lisa_data"
MODEL_DIR="${RAGNET_MODEL_ROOT}/LLaVA-Lightning-7B-v1-1"

mkdir -p "${RAGNET_RAW_ROOT}" "${LISA_ROOT}" "${RAGNET_MODEL_ROOT}"

download() {
    local url="$1"
    local output="$2"

    if [[ -s "${output}" ]]; then
        echo "[skip] ${output}"
        return
    fi
    if [[ "${RAGNET_OFFLINE:-0}" == "1" ]]; then
        echo "Missing archive in offline extraction phase: ${output}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${output}")"
    echo "[download] ${url}"
    curl \
        --fail \
        --location \
        --retry 8 \
        --retry-delay 10 \
        --continue-at - \
        --output "${output}.part" \
        "${url}"
    mv "${output}.part" "${output}"
}

extract_zip_once() {
    local archive="$1"
    local destination="$2"
    local marker="$3"

    if [[ "${RAGNET_FETCH_ONLY:-0}" == "1" ]]; then
        return
    fi
    if [[ -f "${marker}" ]]; then
        echo "[skip] extracted ${archive}"
        return
    fi

    echo "[verify] ${archive}"
    unzip -tqq "${archive}"
    mkdir -p "${destination}"
    echo "[extract] ${archive} -> ${destination}"
    unzip -q -o "${archive}" -d "${destination}"
    touch "${marker}"
}

extract_tar_once() {
    local archive="$1"
    local destination="$2"
    local marker="$3"

    if [[ "${RAGNET_FETCH_ONLY:-0}" == "1" ]]; then
        return
    fi
    if [[ -f "${marker}" ]]; then
        echo "[skip] extracted ${archive}"
        return
    fi

    echo "[verify] ${archive}"
    tar -tf "${archive}" >/dev/null
    mkdir -p "${destination}"
    echo "[extract] ${archive} -> ${destination}"
    tar -xf "${archive}" -C "${destination}"
    touch "${marker}"
}

prepare_base_model() {
    echo "[model] Dongming97/LLaVA-Lightning-7B-v1-1"
    python - "${MODEL_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Dongming97/LLaVA-Lightning-7B-v1-1",
    local_dir=sys.argv[1],
)
PY
}

prepare_llava_annotations() {
    local destination="${LISA_ROOT}/llava_dataset"
    mkdir -p "${destination}"
    echo "[dataset] LLaVA-Instruct-150K annotations"
    python - "${destination}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="liuhaotian/LLaVA-Instruct-150K",
    repo_type="dataset",
    filename="llava_instruct_150k.json",
    local_dir=sys.argv[1],
)
PY
}

prepare_ade20k() {
    local archive="${RAGNET_RAW_ROOT}/ADEChallengeData2016.zip"
    download \
        "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" \
        "${archive}"
    extract_zip_once \
        "${archive}" \
        "${LISA_ROOT}" \
        "${LISA_ROOT}/.ade20k-extracted"
    if [[ -d "${LISA_ROOT}/ADEChallengeData2016" && ! -e "${LISA_ROOT}/ade20k" ]]; then
        mv "${LISA_ROOT}/ADEChallengeData2016" "${LISA_ROOT}/ade20k"
    fi
}

prepare_coco() {
    local train2017="${RAGNET_RAW_ROOT}/train2017.zip"
    local train2014="${RAGNET_RAW_ROOT}/train2014.zip"

    # The official COCO host currently presents a mismatched TLS certificate on
    # TIGRIS, so use the official HTTP endpoints linked by LISA.
    download "http://images.cocodataset.org/zips/train2017.zip" "${train2017}"
    extract_zip_once \
        "${train2017}" \
        "${LISA_ROOT}/coco" \
        "${LISA_ROOT}/coco/.train2017-extracted"

    download "http://images.cocodataset.org/zips/train2014.zip" "${train2014}"
    extract_zip_once \
        "${train2014}" \
        "${LISA_ROOT}/refer_seg/images/mscoco/images" \
        "${LISA_ROOT}/refer_seg/images/mscoco/images/.train2014-extracted"
}

prepare_cocostuff() {
    local archive="${RAGNET_RAW_ROOT}/stuffthingmaps_trainval2017.zip"
    download \
        "https://calvin-vision.net/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip" \
        "${archive}"
    extract_zip_once \
        "${archive}" \
        "${LISA_ROOT}/cocostuff" \
        "${LISA_ROOT}/cocostuff/.stuffthingmaps-extracted"
}

prepare_mapillary() {
    local archive="${RAGNET_RAW_ROOT}/mapillary-vistas-dataset_public_v2.0.zip"
    if [[ ! -s "${archive}" && "${RAGNET_OFFLINE:-0}" != "1" ]]; then
        echo "[dataset] Mapillary Vistas v2.0 (gated; prior terms acceptance required)"
        python - "${RAGNET_RAW_ROOT}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="candylion/mapillary-vistas-v2",
    repo_type="dataset",
    filename="mapillary-vistas-dataset_public_v2.0.zip",
    local_dir=sys.argv[1],
)
PY
    fi
    if [[ ! -s "${archive}" ]]; then
        echo "Missing gated Mapillary archive: ${archive}" >&2
        exit 1
    fi
    extract_zip_once \
        "${archive}" \
        "${LISA_ROOT}/mapillary" \
        "${LISA_ROOT}/mapillary/.v2-extracted"
}

prepare_paco_lvis() {
    local archive="${RAGNET_RAW_ROOT}/paco_lvis_v1.zip"
    download \
        "https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip" \
        "${archive}"
    extract_zip_once \
        "${archive}" \
        "${LISA_ROOT}/vlpart/paco/annotations" \
        "${LISA_ROOT}/vlpart/paco/annotations/.paco-lvis-extracted"
}

prepare_referring_annotations() {
    local name
    local timestamp
    local archive
    for spec in \
        "refcoco 20220413011718" \
        "refcoco+ 20220413011656" \
        "refcocog 20220413012904" \
        "refclef 20220413011817"; do
        read -r name timestamp <<<"${spec}"
        archive="${RAGNET_RAW_ROOT}/${name}.zip"
        download \
            "https://web.archive.org/web/${timestamp}id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/${name}.zip" \
            "${archive}"
        extract_zip_once \
            "${archive}" \
            "${LISA_ROOT}/refer_seg" \
            "${LISA_ROOT}/refer_seg/.${name}-extracted"
    done
}

prepare_refclef_images() {
    local archive="${RAGNET_RAW_ROOT}/saiapr_tc-12.zip"
    if [[ ! -s "${archive}" && "${RAGNET_OFFLINE:-0}" != "1" ]]; then
        # The upstream UNC host is retired and its archived URL frequently
        # stalls. This mirror contains the same archive name referenced by the
        # official REFER README.
        echo "[dataset] RefCLEF saiapr_tc-12 mirror"
        python - "${archive}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path(sys.argv[1])
downloaded = Path(
    hf_hub_download(
        repo_id="cxz0416/saiapr_tc",
        repo_type="dataset",
        filename="saiapr_tc-12.zip",
        local_dir=target.parent,
    )
)
if downloaded.resolve() != target.resolve():
    raise RuntimeError(f"unexpected download path: {downloaded}")
PY
    fi
    if [[ ! -s "${archive}" ]]; then
        echo "Missing archive in offline extraction phase: ${archive}" >&2
        exit 1
    fi
    extract_zip_once \
        "${archive}" \
        "${LISA_ROOT}/refer_seg/images" \
        "${LISA_ROOT}/refer_seg/images/.saiapr-tc12-extracted"
}

prepare_reasonseg() {
    local marker="${LISA_ROOT}/reason_seg/.reasonseg-downloaded"
    mkdir -p "${LISA_ROOT}/reason_seg"
    if [[ -f "${marker}" ]]; then
        echo "[skip] downloaded ReasonSeg"
    elif [[ "${RAGNET_OFFLINE:-0}" == "1" ]]; then
        echo "Missing ReasonSeg in offline extraction phase." >&2
        exit 1
    else
        echo "[dataset] ReasonSeg official Google Drive folder"
        gdown \
            --folder \
            --remaining-ok \
            --continue \
            --output "${LISA_ROOT}/reason_seg/" \
            "https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing"
        touch "${marker}"
    fi

    local split
    for split in train val test; do
        extract_zip_once \
            "${LISA_ROOT}/reason_seg/ReasonSeg/${split}.zip" \
            "${LISA_ROOT}/reason_seg/ReasonSeg" \
            "${LISA_ROOT}/reason_seg/ReasonSeg/.${split}-extracted"
    done
}

prepare_pascal_part_sources() {
    local annotations="${RAGNET_RAW_ROOT}/pascal-part-trainval.tar.gz"
    local voc="${RAGNET_RAW_ROOT}/VOCtrainval_03-May-2010.tar"
    local converter="${RAGNET_RAW_ROOT}/pascal_part_mat2json.py"
    local destination="${LISA_ROOT}/vlpart/pascal_part"

    download \
        "https://raw.githubusercontent.com/facebookresearch/VLPart/main/tools/pascal_part_mat2json.py" \
        "${converter}"
    download "https://roozbehm.info/pascal-parts/trainval.tar.gz" "${annotations}"
    extract_tar_once \
        "${annotations}" \
        "${destination}" \
        "${destination}/.pascal-parts-extracted"

    download \
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar" \
        "${voc}"
    extract_tar_once \
        "${voc}" \
        "${destination}" \
        "${destination}/.voc2010-extracted"

    if [[ "${RAGNET_FETCH_ONLY:-0}" != "1" && ! -s "${destination}/train.json" ]]; then
        echo "[convert] PASCAL-Part annotations -> train.json"
        python "${converter}" \
            --split_path="${destination}/VOCdevkit/VOC2010/ImageSets/Main" \
            --split="train.txt" \
            --ann_path="${destination}/Annotations_Part" \
            --ann_out="${destination}/train.json"
    fi
}

case "${1:-public}" in
    model)
        prepare_base_model
        ;;
    public)
        prepare_base_model
        prepare_llava_annotations
        prepare_ade20k
        prepare_coco
        prepare_cocostuff
        prepare_paco_lvis
        prepare_referring_annotations
        prepare_reasonseg
        prepare_pascal_part_sources
        ;;
    public-fetch)
        export RAGNET_FETCH_ONLY=1
        prepare_base_model
        prepare_llava_annotations
        prepare_ade20k
        prepare_coco
        prepare_cocostuff
        prepare_paco_lvis
        prepare_referring_annotations
        prepare_reasonseg
        prepare_pascal_part_sources
        ;;
    public-extract)
        export RAGNET_OFFLINE=1
        export HF_HUB_OFFLINE=1
        prepare_base_model
        prepare_llava_annotations
        prepare_ade20k
        prepare_coco
        prepare_cocostuff
        prepare_paco_lvis
        prepare_referring_annotations
        prepare_reasonseg
        prepare_pascal_part_sources
        ;;
    refclef-images|refclef-fetch|refclef-extract)
        if [[ "$1" == "refclef-fetch" ]]; then
            export RAGNET_FETCH_ONLY=1
        elif [[ "$1" == "refclef-extract" ]]; then
            export RAGNET_OFFLINE=1
        fi
        prepare_refclef_images
        ;;
    mapillary|mapillary-fetch|mapillary-extract)
        if [[ "$1" == "mapillary-fetch" ]]; then
            export RAGNET_FETCH_ONLY=1
        elif [[ "$1" == "mapillary-extract" ]]; then
            export RAGNET_OFFLINE=1
        fi
        prepare_mapillary
        ;;
    *)
        echo "Usage: $0 {model|public|public-fetch|public-extract|refclef-images|refclef-fetch|refclef-extract|mapillary|mapillary-fetch|mapillary-extract}" >&2
        exit 2
        ;;
esac

echo "[done] training assets prepared under ${LISA_ROOT}"
