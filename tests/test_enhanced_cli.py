import json

import pytest
from sep_dataset import cli,compatible_builder


def test_default_build_routes_original_and_new_outputs_separately(monkeypatch):
    calls=[]
    monkeypatch.setattr(compatible_builder,'build_compatible_dataset',lambda **kw:calls.append(('legacy',kw)))
    monkeypatch.setattr(cli,'_enhance',lambda *args:calls.append(('enhance',args)))
    cli.main(['build'])
    assert calls[0][1]['outdir']==cli.LEGACY_DIR
    assert calls[1][1][:2]==(cli.LEGACY_DIR,cli.ENHANCED_DIR)


def test_legacy_only_retains_explicit_original_output(monkeypatch):
    calls=[]
    monkeypatch.setattr(compatible_builder,'build_compatible_dataset',lambda **kw:calls.append(kw))
    monkeypatch.setattr(cli,'_enhance',lambda *args:pytest.fail('No enhancement requested'))
    cli.main(['build','--legacy-only','--outdir','original'])
    assert calls[0]['outdir']=='original'


def test_failed_partial_build_does_not_silently_generate_partial_paper_figures(tmp_path):
    (tmp_path/'build_status.json').write_text(json.dumps({'state':'failed','error':'Example native archive failure'}))
    with pytest.raises(RuntimeError,match='Example native archive failure'):
        cli.main(['plot','--dataset-dir',str(tmp_path)])
    with pytest.raises(RuntimeError,match='incomplete'):
        cli.main(['paper-plot','--dataset-dir',str(tmp_path)])
