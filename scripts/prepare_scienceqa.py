"""Prepare a pinned, seeded ScienceQA image-test subset; publish identifiers only."""
import base64
import hashlib
import json
import random
import urllib.request
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
GIT_REV = '2cbf8318e07b9ece895bb2ae605e71e38d623264'
HF_REV = 'f18b0a70359ebfb41f658fd564208d0355b013f4'
PARQUET = 'data/test-00000-of-00001-f0e719df791966ff.parquet'

def fingerprint(row):
    return json.dumps([row[k] for k in ('question','choices','answer','hint','task','grade','subject','topic','category','skill','lecture','solution')],ensure_ascii=False)

def main():
    folder = ROOT/'data/scienceqa'
    folder.mkdir(parents=True,exist_ok=True)
    metadata = folder/'problems.json'
    if not metadata.exists():
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/lupantech/ScienceQA/{GIT_REV}/data/scienceqa/problems.json',metadata)
    assert hashlib.sha256(metadata.read_bytes()).hexdigest()=='4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7', 'Unexpected metadata bytes'
    official = json.loads(metadata.read_text())
    official_test=[(qid,row) for qid,row in official.items() if row['split']=='test']
    local_parquet=folder/'test.parquet'
    path = local_parquet if local_parquet.exists() else hf_hub_download('derek-thomas/ScienceQA',PARQUET,repo_type='dataset',revision=HF_REV)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()=='235e92d1f30155266df76bc9f28fc6e4fcb6bec2c6a8c7d67f9086ea6b392a84', 'Unexpected parquet bytes'
    records = pq.read_table(path).to_pylist()
    assert len(records)==len(official_test)==4241
    pool = {}
    # The author-linked exporter preserves problems.json insertion order.
    # Verify every field of every row before associating the original IDs.
    for (qid,source),row in zip(official_test,records):
        assert fingerprint(row)==fingerprint(source), f'Metadata mismatch: {qid}'
        assert bool(row['image'])==bool(source['image'])
        if row['image']:
            assert row['image']['bytes']
            pool[qid]=row
    ids = sorted(random.Random(42).sample(sorted(pool,key=int),100),key=int)
    rows=[]
    manifest_rows=[]
    for qid in ids:
        row=pool[qid]
        image=row['image']['bytes']
        keys=[f'option_{i}' for i in range(len(row['choices']))]
        question=row['question']
        if row['hint']:
            question += '\nContext: '+row['hint']
        rows.append(dict(id=qid,family=row['subject'],expected=keys[row['answer']],request=dict(type='choice',question=question,options=dict(zip(keys,row['choices'])),state={},images=[dict(base64=base64.b64encode(image).decode(),timestamp_ms=0,camera_id='question')])) )
        manifest_rows.append(dict(id=qid,expected=keys[row['answer']],subject=row['subject'],image_sha256=hashlib.sha256(image).hexdigest(),metadata_sha256=hashlib.sha256(fingerprint(row).encode()).hexdigest()))
    dataset=dict(name='ScienceQA',split='test',selection='100 image-bearing questions sampled with random.Random(42) from numeric-sorted IDs',seed=42,count=100,image_test_pool_count=len(pool),official_revision=GIT_REV,hf_revision=HF_REV,parquet_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),license='CC-BY-NC-SA-4.0',source='https://github.com/lupantech/ScienceQA')
    suite=dict(suite_id='scienceqa_image_test_100_seed42',scope='ScienceQA_official_test_fixed_100_image_question_subset',dataset=dataset,rows=rows)
    payload=json.dumps(suite,ensure_ascii=False,indent=2).encode()
    (folder/'suite.json').write_bytes(payload)
    manifest=dict(suite_id=suite['suite_id'],suite_sha256=hashlib.sha256(payload).hexdigest(),dataset=dataset,rows=manifest_rows)
    (ROOT/'benchmarks/scienceqa-test-100-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dataset),flush=True)

if __name__=='__main__': main()
