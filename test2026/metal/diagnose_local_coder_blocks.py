#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Secondary contract diagnosis of one Python fence; never replaces strict scores."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from evaluate_local_coder_quality import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--strict-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    strict = json.loads(args.strict_report.read_text())
    digest = hashlib.sha256(args.suite.read_bytes()).hexdigest()
    if strict['suite_sha256'] != digest or strict['total'] != len(suite['tasks']):
        raise ValueError('Strict report/suite mismatch')
    results = []
    for task in suite['tasks']:
        path = args.responses/task['id']/'request-0.result.json'
        row = dict(task=task['id'], status='missing_response')
        if path.exists():
            response = json.loads(path.read_text())
            row['response_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if response.get('status') != 'completed':
                row['status'] = 'generation_failed'
            elif response.get('finish_reason') == 'length':
                row['status'] = 'truncated'
            else:
                text = response['text']
                # Multiple blocks may be examples, competing implementations,
                # or auxiliary code. Do not guess which to execute.
                matches = re.findall(r'(?m)^```(?:python|py)\s*\n(.*?)^```\s*$', text, re.S)
                if len(matches) != 1 or len(re.findall(r'(?m)^```', text)) != 2:
                    row['status'] = 'no_unique_python_fence'
                else:
                    code = matches[0]
                    row.update(execute(code, task['tests']))
                    row['extracted_code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
        results.append(row)
    report = dict(diagnostic_only=True, strict_passed=strict['passed'], strict_total=strict['total'],
                  extracted_contract_passed=sum(row['status'] == 'passed' for row in results),
                  results=results, suite_sha256=digest,
                  strict_report_sha256=hashlib.sha256(args.strict_report.read_bytes()).hexdigest(),
                  scope='Single-fence contract diagnosis; surrounding prose still fails the unchanged code-only gate')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('strict_passed', 'strict_total', 'extracted_contract_passed', 'diagnostic_only')}))


if __name__ == '__main__':
    main()
