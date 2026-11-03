"""과목: 정답표, 명렬표, 신경망, 예시 답안지 (노트북 5-2의 SUBJECTS)

data/subjects/과목/ 폴더마다 표 파일(CSV, 엑셀)에서 정답표와 명렬표를 찾고, sheets/의 사진을 예시 답안지로 쓴다.
신경망은 그 정답표로 시뮬레이션 데이터를 만들어 학습한다(과목당 1초 안팎). 같은 정답표면 저장해 둔 모델을 다시 쓴다.
폴더에 노트북에서 내보낸 grader.json이 있고 정답표 지문이 같으면 그 신경망을 쓴다.
"""
import json
import logging
import os
from dataclasses import dataclass, field

from grader.answer_key import fingerprint
from grader.model import GraderModel
from grader.simulate import make_dataset
from grader.tables import ANSWER_KEY, ROSTER, TABLE_EXT, classify, parse_answer_key, parse_roster, read_table_file
from grader.trainer import train
from vision.layout import ANSWER_BOXES
from vision.pages import expand

log = logging.getLogger('demo')


@dataclass
class Subject:
    name: str
    answer_key: dict
    roster: dict | None
    model: GraderModel
    model_info: dict                                    # 웹페이지에 보이는 신경망 출처와 검증 정확도
    samples: list = field(default_factory=list)         # [(시험지, 제목, 답안지 이미지 경로)]

    @property
    def questions(self):
        """시험지에 칸이 있는 문항만 사진으로 채점할 수 있다"""
        return [q for q in self.answer_key if q in ANSWER_BOXES]


def trained_model(answer_key, settings):
    """정답표로 학습한 신경망. 같은 정답표, 같은 학습 설정이면 저장해 둔 것을 쓴다"""
    fp = fingerprint(answer_key)
    path = settings.work_dir / 'models' / f'{fp}_s{settings.seed}_n{settings.n_samples}.json'
    if path.exists():
        model = GraderModel.load(path)
    else:
        X, y, _ = make_dataset(answer_key, settings.n_samples, settings.seed)
        model = train(X, y, seed=settings.seed)
        model.info.update(answer_key=fp, seed=settings.seed, samples=settings.n_samples)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
    return model, {'source': 'trained_for_answer_key', 'val_accuracy': model.info.get('val_accuracy'),
                   'epochs': model.info.get('epochs'), 'answer_key': fp}


def uploaded_model(model, answer_key):
    """노트북에서 내보내 올린 grader.json → (모델, 출처 정보, 경고)"""
    fp = fingerprint(answer_key)
    info = {'source': 'notebook_upload', 'val_accuracy': model.info.get('val_accuracy'), 'epochs': model.info.get('epochs'),
            'answer_key': model.info.get('answer_key')}
    warn = [] if model.info.get('answer_key') in (None, fp) else [
        f"올린 grader.json은 다른 정답표(지문 {model.info.get('answer_key')})로 학습한 신경망입니다. 지금 정답표의 지문은 {fp}입니다"]
    return model, info, warn


def load_subject(folder, settings):
    name = os.path.basename(folder)
    tables = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(TABLE_EXT) and not f.startswith('.'):
            tables += read_table_file(os.path.join(folder, f))
    found, _ = classify(tables)
    if ANSWER_KEY not in found:
        raise ValueError('정답표가 없습니다')
    answer_key = parse_answer_key(found[ANSWER_KEY])
    roster = parse_roster(found[ROSTER]) if ROSTER in found else None

    model = info = None
    exported = os.path.join(folder, 'grader.json')
    if os.path.exists(exported):   # 노트북 3-7에서 내보낸 신경망
        m = GraderModel.from_dict(json.load(open(exported, encoding='utf-8')))
        if m.info.get('answer_key') == fingerprint(answer_key):
            model, info = m, {'source': 'notebook', 'val_accuracy': m.info.get('val_accuracy'),
                              'epochs': m.info.get('epochs'), 'answer_key': m.info.get('answer_key')}
        else:
            log.warning('%s: grader.json의 정답표 지문이 달라 쓰지 않고 다시 학습합니다', name)
    if model is None:
        model, info = trained_model(answer_key, settings)

    subject = Subject(name, answer_key, roster, model, info)
    sheet_dir = os.path.join(folder, 'sheets')
    if os.path.isdir(sheet_dir):
        files = sorted(os.path.join(sheet_dir, f) for f in os.listdir(sheet_dir))
        pages, _ = expand(files, str(settings.work_dir / 'pages' / name), str(settings.work_dir / '_zip'))
        # 과목이 달라도 학생 이름이 같을 수 있어 과목 이름을 붙임 (노트북 5-2와 같음)
        subject.samples = [(f'{name}_{stem}', stem, path) for stem, path in pages]
    return subject


def load_subjects(settings):
    """data/subjects/의 과목들 → {과목: Subject}. 기본 과목이 맨 앞 (웹페이지는 첫 과목을 먼저 고름)"""
    root = settings.data_dir / 'subjects'
    subjects = {}
    for name in sorted(os.listdir(root)) if root.is_dir() else []:
        folder = root / name
        if not folder.is_dir() or name.startswith(('.', '_')):
            continue
        try:
            subjects[name] = load_subject(str(folder), settings)
        except Exception as e:
            log.warning('%s: 과목 자료를 읽지 못해 건너뜀 (%s)', name, e)
            continue
        s = subjects[name]
        log.info('%s: 정답표 %d문항, 예시 답안지 %d장, 신경망 %s (검증 정확도 %s)', name, len(s.answer_key), len(s.samples),
                 s.model_info['source'], s.model_info.get('val_accuracy'))
    order = sorted(subjects, key=lambda n: n != settings.default_subject)
    return {n: subjects[n] for n in order}
