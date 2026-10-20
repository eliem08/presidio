from pathlib import Path

import presidio_analyzer
import pytest
import spacy
import yaml
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers import FrRppsRecognizer
from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

from tests.mocks import NlpEngineMock

ENTITY = "FR_RPPS"
PATTERN_SCORE = 0.1


@pytest.fixture(scope="module")
def recognizer():
    """Recognizer under test."""
    return FrRppsRecognizer()


def _french_artifacts(text):
    doc = spacy.blank("fr")(text)
    return NlpArtifacts(
        entities=[],
        tokens=doc,
        tokens_indices=[token.idx for token in doc],
        lemmas=[token.lower_ for token in doc],
        nlp_engine=NlpEngineMock(),
        language="fr",
    )


@pytest.mark.parametrize(
    "text, start, end",
    [
        ("10001234565", 0, 11),
        ("Dr Martin, RPPS 10123456781", 16, 27),
        ("Ordonnance établie par le médecin n° 10999999997.", 37, 48),
    ],
)
def test_when_valid_number_then_detected_with_pattern_score(
    recognizer, text, start, end
):
    """A valid number keeps the pattern score: the checksum only rejects."""
    results = recognizer.analyze(text, [ENTITY])

    assert [(r.entity_type, r.start, r.end) for r in results] == [(ENTITY, start, end)]
    assert results[0].score == pytest.approx(PATTERN_SCORE)


@pytest.mark.parametrize(
    "text",
    [
        "10001234566",  # wrong check digit
        "20001234561",  # does not start with 10
        "1000123456",  # 10 digits
        "100012345650",  # 12 digits
        "AB10001234565",  # embedded in a token
        "Référence 10001234560 du colis",  # lookalike reference, invalid check digit
    ],
)
def test_when_invalid_or_lookalike_then_not_detected(recognizer, text):
    """Wrong check digit, wrong length or embedded digits are not reported."""
    assert recognizer.analyze(text, [ENTITY]) == []


@pytest.mark.parametrize(
    "number, expected",
    [
        ("10001234565", None),
        ("10123456781", None),
        ("10001234566", False),
        ("1000123456", False),
        ("1000123456A", False),
    ],
)
def test_validate_result_only_rejects(recognizer, number, expected):
    """Valid numbers return None (no promotion), invalid ones return False."""
    assert recognizer.validate_result(number) is expected


def test_when_context_word_precedes_then_score_increases(recognizer):
    """A French context word before the number raises the score."""
    enhancer = LemmaContextAwareEnhancer()
    scores = []
    for text in (
        "Le code 10001234565 figure au dossier.",
        "Le numéro RPPS 10001234565 figure au dossier.",
    ):
        results = recognizer.analyze(text, [ENTITY])
        enhanced = enhancer.enhance_using_context(
            text, results, _french_artifacts(text), [recognizer]
        )
        scores.append(enhanced[0].score)

    assert scores[0] == pytest.approx(PATTERN_SCORE)
    assert scores[1] == pytest.approx(max(PATTERN_SCORE + 0.35, 0.4))


def test_loads_and_detects_when_enabled_in_yaml():
    """Detection works through the path users configure: the shipped YAML entry.

    The recognizer only supports ``fr``, so the top-level ``supported_languages``
    must include ``fr`` or the registry silently loads nothing.
    """
    conf = Path(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    entries = [
        r
        for r in yaml.safe_load(conf.read_text(encoding="utf-8"))["recognizers"]
        if r.get("name") == "FrRppsRecognizer"
    ]
    assert len(entries) == 1
    assert entries[0]["country_code"] == "fr"
    assert entries[0]["enabled"] is False

    registry = RecognizerRegistryProvider(
        registry_configuration={
            "supported_languages": ["fr"],
            "recognizers": [dict(entries[0], enabled=True)],
        }
    ).create_recognizer_registry()

    assert [type(r).__name__ for r in registry.recognizers] == ["FrRppsRecognizer"]
    loaded = registry.recognizers[0]
    results = loaded.analyze("Le numéro RPPS 10001234565 figure au dossier.", [ENTITY])
    assert [(r.entity_type, r.score) for r in results] == [
        (ENTITY, pytest.approx(PATTERN_SCORE))
    ]
