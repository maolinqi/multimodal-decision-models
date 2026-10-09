"""Paired native-generation vs decision-forward probes on identical prepared inputs.

This checks inference-path retention, not full base-model capability retention.
No model parameters are trained; the baseline uses the same checkpoint, preprocessing,
prompt, attention backend and precision, through the official generate implementation.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from multimodal_decision.registry import create_model, MODELS


def official_first_step(actor, inputs):
    kwargs = dict(max_new_tokens=1, do_sample=False, return_dict_in_generate=True,
                  output_logits=True, output_scores=True)
    if getattr(actor, 'family', None) == 'minicpm':
        output = actor.model.generate(**inputs, tokenizer=actor.tokenizer,
                                      decode_text=False, **kwargs)
    else:
        output = actor.model.generate(**inputs, **kwargs)
    # Raw logits exclude generation processors: compare the model, not decoder policy.
    return output.logits[0][0].float()


def snapshot_versions(model):
    return {name: (id(parameter), parameter._version) for name, parameter in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(MODELS))
    parser.add_argument('--suite', default='benchmarks/base_retention_v1.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    suite_bytes = Path(args.suite).read_bytes()
    suite = json.loads(suite_bytes)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    print('LOADING', args.model, flush=True)
    actor = create_model(args.model)
    if actor.adapter:
        raise ValueError('This comparison requires the unchanged base checkpoint, without LoRA')
    versions = snapshot_versions(actor.model)
    results = []
    with torch.inference_mode():
        for row in suite['rows']:
            inputs, keys, _ = actor.prepare(row['request'])
            torch.cuda.synchronize()
            started = time.perf_counter()
            direct = actor.logits(inputs)
            torch.cuda.synchronize()
            decision_ms = (time.perf_counter()-started)*1000
            native = official_first_step(actor, inputs)
            label_ids = actor.label_ids[:len(keys)]
            native_prob = torch.softmax(native[label_ids], dim=0)
            decision_prob = torch.softmax(direct[label_ids], dim=0)
            finite = torch.isfinite(native) & torch.isfinite(direct)
            nonfinite_matches = bool(torch.equal(torch.isfinite(native), torch.isfinite(direct)))
            full_delta = float((native[finite]-direct[finite]).abs().max())
            candidate_delta = float((native[label_ids]-direct[label_ids]).abs().max())
            probability_delta = float((native_prob-decision_prob).abs().max())
            native_answer = keys[int(native_prob.argmax())]
            decision_answer = keys[int(decision_prob.argmax())]
            record = dict(id=row['id'], family=row['family'], expected=row['expected'],
                          native_answer=native_answer, decision_answer=decision_answer,
                          native_correct=native_answer==row['expected'],
                          decision_correct=decision_answer==row['expected'],
                          decision_agrees=native_answer==decision_answer,
                          native_distribution=dict(zip(keys,native_prob.cpu().tolist())),
                          decision_distribution=dict(zip(keys,decision_prob.cpu().tolist())),
                          full_vocab_max_abs_diff=full_delta,
                          candidate_max_abs_diff=candidate_delta,
                          probability_max_abs_diff=probability_delta,
                          finite_pattern_matches=nonfinite_matches,
                          input_tokens=int(inputs['input_ids'].shape[-1]),
                          visual_tokens=actor.visual_tokens(inputs),
                          image_count=len(row['request']['images']), decision_forward_ms=decision_ms)
            if row['request']['images'] and record['visual_tokens'] <= 0:
                raise AssertionError('Missing visual token path')
            results.append(record)
            # Partial progress is evidence, never treated as a completed comparison.
            output.with_suffix('.partial.json').write_text(json.dumps(dict(status='running', model_id=args.model, results=results),ensure_ascii=False,indent=2))
            print('PAIRED', row['id'], native_answer, decision_answer, full_delta, flush=True)
    after = snapshot_versions(actor.model)
    changed = sum(versions.get(name)!=value for name,value in after.items()) + len(set(versions)-set(after))
    families = sorted({row['family'] for row in results})
    grouped = {}
    for family in families:
        selected = [row for row in results if row['family']==family]
        grouped[family] = dict(count=len(selected),
                               native_correct=sum(row['native_correct'] for row in selected),
                               decision_correct=sum(row['decision_correct'] for row in selected),
                               agreements=sum(row['decision_agrees'] for row in selected))
    passed = changed==0 and all(row['decision_agrees'] and row['finite_pattern_matches']
                               and row['candidate_max_abs_diff']<=1e-3
                               and row['full_vocab_max_abs_diff']<=1e-3 for row in results)
    summary = dict(count=len(results),native_correct=sum(row['native_correct'] for row in results),
                   decision_correct=sum(row['decision_correct'] for row in results),
                   agreements=sum(row['decision_agrees'] for row in results),
                   max_full_vocab_abs_diff=max(row['full_vocab_max_abs_diff'] for row in results),
                   max_candidate_abs_diff=max(row['candidate_max_abs_diff'] for row in results),
                   max_probability_abs_diff=max(row['probability_max_abs_diff'] for row in results),
                   changed_parameter_version_count=changed, by_family=grouped)
    report = dict(model_id=args.model,base_model=MODELS[args.model][1],
                  status='completed',paired_path_retention_passed=passed,
                  suite_id=suite['suite_id'],suite_sha256=hashlib.sha256(suite_bytes).hexdigest(),
                  baseline='official_generate_first_step_raw_logits',
                  controlled_variables=['checkpoint','prepared_input','prompt','precision','attention_backend','candidate_set'],
                  scope=suite.get('scope','fixed_text_and_synthetic_visual_probes_with_bounded_preprocessing'),
                  dataset=suite.get('dataset'),
                  full_base_capability_retention_proven=False,training_applied=False,
                  real_world_generalization_proven=False,summary=summary,results=results,
                  torch_version=torch.__version__,transformers_version=__import__('transformers').__version__)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    output.with_suffix('.partial.json').unlink(missing_ok=True)
    print('SUMMARY',json.dumps(summary),flush=True)
    if not passed:
        raise SystemExit('Paired inference-path retention check failed; negative evidence saved')


if __name__ == '__main__':
    main()
