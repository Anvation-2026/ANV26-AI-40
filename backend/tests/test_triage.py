from backend.schemas.analysis import AnalysisStatus
from backend.services.triage_service import evaluate_triage


def test_triage_precedence_poor_quality_beats_all(fake_ml_doubles, fake_model_meta):
    # ML output has poor quality AND ood=True AND high uncertainty
    res = fake_ml_doubles.poor_quality()
    res.ood.is_ood = True
    res.uncertainty.level = "high"

    analysis = evaluate_triage(res, fake_model_meta, "req-1", mode="real")
    assert analysis.status == AnalysisStatus.POOR_QUALITY
    assert analysis.triage.action == "resubmit_better_sample"
    assert analysis.finding is None
    assert analysis.probability is None
    assert analysis.heatmap.available is False


def test_triage_precedence_ood_beats_uncertain_and_success(fake_ml_doubles, fake_model_meta):
    res = fake_ml_doubles.ood()
    res.uncertainty.level = "high"
    res.finding = "pneumonia"
    res.calibrated_probability = 0.90

    analysis = evaluate_triage(res, fake_model_meta, "req-2", mode="real")
    assert analysis.status == AnalysisStatus.OOD
    assert analysis.triage.action == "unsupported_input"
    assert "educational chest X-rays only" in analysis.triage.message
    assert analysis.finding is None
    assert analysis.probability is None
    assert analysis.heatmap.available is False


def test_triage_high_uncertainty_forces_uncertain(fake_ml_doubles, fake_model_meta):
    res = fake_ml_doubles.high_uncertainty()
    analysis = evaluate_triage(res, fake_model_meta, "req-3", mode="real")
    assert analysis.status == AnalysisStatus.UNCERTAIN
    assert analysis.triage.action == "expert_review_required_abstained"
    assert analysis.finding is None  # Must suppress finding
    assert analysis.probability is None  # Must suppress probability
    assert analysis.heatmap.available is False


def test_triage_model_abstained_forces_uncertain(fake_ml_doubles, fake_model_meta):
    res = fake_ml_doubles.abstained_by_model()
    analysis = evaluate_triage(res, fake_model_meta, "req-4", mode="real")
    assert analysis.status == AnalysisStatus.UNCERTAIN
    assert analysis.finding is None
    assert analysis.probability is None
    assert analysis.heatmap.available is False


def test_triage_success_pneumonia_passes_values_exact(fake_ml_doubles, fake_model_meta):
    res = fake_ml_doubles.pneumonia_success()
    analysis = evaluate_triage(res, fake_model_meta, "req-5", mode="real")
    assert analysis.status == AnalysisStatus.SUCCESS
    assert analysis.finding == "pneumonia"
    assert analysis.raw_score == 0.884
    # Exact equality asserted with fake ML value
    assert analysis.probability == 0.871
    assert analysis.heatmap.available is True
    assert analysis.heatmap.data_url.startswith("data:image/png;base64,")


def test_triage_success_normal_passes_values_exact(fake_ml_doubles, fake_model_meta):
    res = fake_ml_doubles.normal_success()
    analysis = evaluate_triage(res, fake_model_meta, "req-6", mode="real")
    assert analysis.status == AnalysisStatus.SUCCESS
    assert analysis.finding == "normal"
    assert analysis.raw_score == 0.082
    assert analysis.probability == 0.915
    assert analysis.heatmap.available is True


def test_triage_missing_heatmap_handled(fake_ml_doubles, fake_model_meta):
    res = fake_ml_doubles.success_missing_heatmap()
    analysis = evaluate_triage(res, fake_model_meta, "req-7", mode="real")
    assert analysis.status == AnalysisStatus.SUCCESS
    assert analysis.heatmap.available is False
    assert "Heatmap unavailable" in (analysis.heatmap.message or "")


def test_triage_not_evaluated_checks_adds_reasons(fake_ml_doubles, fake_model_meta):
    res = fake_ml_doubles.not_evaluated_checks()
    analysis = evaluate_triage(res, fake_model_meta, "req-8", mode="real")
    assert analysis.status == AnalysisStatus.SUCCESS
    assert any("Image quality was not evaluated" in r for r in analysis.triage.reasons)
    assert any("OOD check was not evaluated" in r for r in analysis.triage.reasons)
