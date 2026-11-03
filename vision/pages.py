"""올린 파일 → 답안지 한 장씩 (노트북 4-1)

사진(jpg, png, 아이폰 heic), 여러 쪽 PDF(쪽마다 답안지 한 장), ZIP(사진과 PDF 묶음)을 받는다.
서버는 다른 사람이 올린 파일을 다루므로, 노트북에 없던 상한(ZIP 안 파일 수와 크기, PDF 쪽수)을 적용한다.
"""
import os
import shutil
import zipfile

from PIL import Image, ImageOps

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.bmp', '.tif', '.tiff')
MAX_ZIP_FILES, MAX_ZIP_BYTES, MAX_PDF_PAGES = 300, 300 * 1024 * 1024, 60


def extract_zip(path, dest):
    """ZIP 풀기 (한글 파일 이름 깨짐 방지: 윈도우 압축(CP949)과 UTF-8 모두 처리, 상위 폴더로 나가는 이름은 무시)"""
    with zipfile.ZipFile(path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > MAX_ZIP_FILES or sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
            raise ValueError(f'ZIP이 너무 큽니다 (파일 {MAX_ZIP_FILES}개, 풀었을 때 {MAX_ZIP_BYTES // 1024 // 1024}MB까지)')
        for info in infos:
            name = info.filename
            if not info.flag_bits & 0x800:   # UTF-8 표시가 없는 이름은 원래 인코딩으로 다시 읽기
                raw = name.encode('cp437', errors='ignore')
                for enc in ('utf-8', 'cp949'):
                    try:
                        name = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        pass
            if '__MACOSX' in name:
                continue
            out = os.path.join(dest, *[p for p in name.replace('\\', '/').split('/') if p not in ('', '.', '..')])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(info) as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst)


def open_image(path):
    """어떤 이미지든 RGB로 열기 (아이폰 HEIC 포함, 사진 회전 정보 반영)"""
    if path.lower().endswith(('.heic', '.heif')):
        import pillow_heif
        pillow_heif.register_heif_opener()
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def _save_image(path, out_dir, stem):
    """jpg, png는 회전 정보가 없으면 그대로 복사하고(용량 절약), 나머지는 PNG로 바꿔 저장"""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg', '.png'):
        with Image.open(path) as im:
            plain = im.getexif().get(0x0112, 1) == 1 and im.mode in ('RGB', 'L')
        if plain:
            out = os.path.join(out_dir, stem + ext)
            shutil.copyfile(path, out)
            return out
    out = os.path.join(out_dir, stem + '.png')
    open_image(path).save(out)
    return out


def expand(paths, out_dir, work_dir=None, _taken=None):
    """사진, PDF(여러 쪽), ZIP → 답안지 한 장씩 [(시험지 이름, 이미지 경로)], 건너뛴 파일 설명.
    이름이 겹치면(ZIP 안 다른 폴더의 같은 이름 등) 저장하기 전에 뒤에 _2, _3을 붙인다"""
    os.makedirs(out_dir, exist_ok=True)
    taken = set() if _taken is None else _taken

    def unique(name):
        cand, k = name, 1
        while cand in taken:
            k += 1
            cand = f'{name}_{k}'
        taken.add(cand)
        return cand

    pages, skipped = [], []
    for p in paths:
        stem, ext = os.path.splitext(os.path.basename(p))
        ext = ext.lower()
        if stem.startswith('.'):
            continue
        if ext == '.zip':
            sub = os.path.join(work_dir or out_dir, '_zip', stem)
            extract_zip(p, sub)
            inner = sorted(os.path.join(r, f) for r, _, fs in os.walk(sub) for f in fs if not f.startswith('.'))
            inner_pages, inner_skipped = expand(inner, out_dir, work_dir, taken)
            pages += inner_pages
            skipped += inner_skipped
        elif ext == '.pdf':
            import pymupdf
            with pymupdf.open(p) as doc:
                if len(doc) > MAX_PDF_PAGES:
                    raise ValueError(f'{os.path.basename(p)}: PDF는 {MAX_PDF_PAGES}쪽까지 받습니다')
                for i, page in enumerate(doc):
                    name = unique(f'{stem}_p{i + 1:02d}' if len(doc) > 1 else stem)
                    out = os.path.join(out_dir, f'{name}.png')
                    page.get_pixmap(dpi=200).save(out)
                    pages.append((name, out))
        elif ext in IMAGE_EXT:
            try:
                name = unique(stem)
                pages.append((name, _save_image(p, out_dir, name)))
            except Exception as e:
                skipped.append(f'{os.path.basename(p)}: 이미지를 열 수 없음 ({type(e).__name__})')
        else:
            skipped.append(f'{os.path.basename(p)}: 답안지 형식이 아니라 건너뜀')
    return pages, skipped
