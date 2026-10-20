from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class FrFinessRecognizer(PatternRecognizer):
    """
    Recognize French FINESS numbers (healthcare and social facility identifiers).

    The FINESS number (Fichier National des Établissements Sanitaires et
    Sociaux) identifies hospitals, clinics, pharmacies and other healthcare or
    social establishments, and the legal entities that run them. It has
    9 characters:
    - 2 characters: département code
    - 6 digits: sequence number
    - 1 digit: check digit using the Luhn algorithm

    Only numeric département codes are covered; Corsican numbers starting with
    ``2A`` or ``2B`` are not detected.

    Reference:
    https://finess.esante.gouv.fr/

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    COUNTRY_CODE = "fr"

    FINESS_LENGTH = 9

    PATTERNS = [
        Pattern(
            "French FINESS number (very weak)",
            r"\b\d{9}\b",
            0.05,
        ),
    ]

    CONTEXT = [
        "finess",
        "n° finess",
        "numéro finess",
        "finess juridique",
        "finess géographique",
        "établissement",
        "etablissement",
        "hôpital",
        "hopital",
        "clinique",
        "pharmacie",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fr",
        supported_entity: str = "FR_FINESS",
        name: Optional[str] = None,
    ):
        patterns = self.PATTERNS if patterns is None else patterns
        context = self.CONTEXT if context is None else context
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=name,
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        """
        Reject numbers that cannot be FINESS numbers.

        A Luhn check digit passes for 1 in 10 arbitrary 9-digit numbers, so a
        valid checksum keeps the pattern score (context words raise it) instead of
        promoting the match to full confidence; an invalid one discards the match.

        :param pattern_text: the text detected by the regex engine
        :return: False if the number is not a valid FINESS number, None otherwise
        """
        if len(pattern_text) != self.FINESS_LENGTH or not pattern_text.isdigit():
            return False

        # "00" is not a département code
        if pattern_text.startswith("00"):
            return False

        return None if self._is_luhn_valid(pattern_text) else False

    @staticmethod
    def _is_luhn_valid(value: str) -> bool:
        digits = [int(digit) for digit in value]
        checksum = 0
        parity = len(digits) % 2

        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit

        return checksum % 10 == 0
