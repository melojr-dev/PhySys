#!/usr/bin/env python3
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def run(label, cmd, cwd=ROOT):
    print(f"\n== {label} ==")
    subprocess.run(cmd, cwd=cwd, check=True)

def clean():
    for p in ROOT.rglob('*.pyc'): p.unlink(missing_ok=True)
    for p in sorted(ROOT.rglob('__pycache__'), reverse=True):
        try: p.rmdir()
        except OSError: pass
    (ROOT/'frontend/tsconfig.tsbuildinfo').unlink(missing_ok=True)

def contract():
    main=(ROOT/'frontend/src/main.tsx').read_text(encoding='utf-8')
    api=(ROOT/'frontend/src/api.ts').read_text(encoding='utf-8')
    css=(ROOT/'frontend/src/styles.css').read_text(encoding='utf-8')
    components=ROOT/'frontend/src/components'
    checks={
        'version': (ROOT/'VERSION').read_text().strip()=='2.21.0-rc.4',
        'phase23 doc': (ROOT/'docs/PHASE23_VALIDATION.md').is_file(),
        'workspace nav': (components/'WorkspaceNav.tsx').is_file() and 'WorkspaceNav' in main,
        'workspace hub': (components/'WorkspaceHub.tsx').is_file() and 'WorkspaceHub' in main,
        'study management': (components/'StudyManagementPanel.tsx').is_file(),
        'formal validation UI': (components/'FormalValidationPanel.tsx').is_file(),
        'research UI': (components/'ResearchWorkspace.tsx').is_file(),
        'admin UI': (components/'AdminWorkspace.tsx').is_file(),
        'gait UI': (components/'GaitPanel.tsx').is_file(),
        'formal run API': all(x in api for x in ['formalReadiness','createFormalRun','formalRuns','closeFormalRun','downloadFormalRun']),
        'session API': all(x in api for x in ['session(jobId','datasetStats','systemMetrics']),
        'publication API retained': 'downloadPublicationPackage' in api,
        'mobile bottom nav': '@media (max-width: 720px)' in css and 'bottom: 0' in css,
        'protocol unchanged': 'PROTOCOL_VERSION = "1.4"' in (ROOT/'backend/src/research/provenance.py').read_text(encoding='utf-8'),
    }
    bad=[k for k,v in checks.items() if not v]
    for k,v in checks.items(): print(('PASS' if v else 'FAIL'), k)
    if bad: raise SystemExit('Fase 23 incompleta: '+', '.join(bad))

def main():
    contract()
    run('Clinical golden baseline',[sys.executable,'scripts/verify_clinical_baseline.py'])
    run('Python compileall',[sys.executable,'-m','compileall','-q','backend','scripts'])
    clean()
    run('Backend regression',[sys.executable,'-m','unittest','discover','-s','tests','-q'],cwd=ROOT/'backend')
    clean()
    tsc=shutil.which('tsc')
    if tsc: run('Frontend TypeScript syntax',[tsc,'--noEmit','--noCheck'],cwd=ROOT/'frontend')
    npm=shutil.which('npm')
    if npm and (ROOT/'frontend/node_modules').is_dir(): run('Frontend build',[npm,'run','build'],cwd=ROOT/'frontend')
    else: print('\nSKIP frontend build completo: node_modules indisponível.')
    clean()
    run('Release manifest',[sys.executable,'scripts/release_manifest.py'])
    run('Release integrity',[sys.executable,'scripts/verify_release.py'])
    run('Static integration doctor',[sys.executable,'scripts/doctor.py'])
    clean()
    return 0
if __name__=='__main__': raise SystemExit(main())
