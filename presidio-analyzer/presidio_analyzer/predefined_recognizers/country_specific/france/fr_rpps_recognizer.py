from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class FrRppsRecognizer(PatternRecognizer):
    """
    Recognize French RPPS numbers (healthcare professional identifiers).

    The RPPS number (Répertoire Partagé des Professionnels de Santé) is the
    national identifier of physicians, pharmacists, midwives, dentists and
    other healthcare professionals in France. It appears on prescriptions and
    in clinical documents, which makes it a direct identifier of the treating
    professional. It has 11 digits, starts with ``10``, and its last digit is a
    check digit using the Luhn algorithm.

    Reference:
    https://esante.gouv.fr/produits-services/repertoire-rpps

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    COUNTRY_CODE = "fr"

    RPPS_LENGTH = 11

    PATTERNS = [
        Pattern(
            "French RPPS number (weak)",
            r"\b10\d{9}\b",
            0.1,
        ),
    ]

    CONTEXT = [
        "rpps",
        "n° rpps",
        "numéro rpps",
        "identifiant rpps",
        "médecin",
        "medecin",
        "docteur",
        "pharmacien",
        "prescripteur",
        "ordonnance",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fr",
        supported_entity: str = "FR_RPPS",
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
        Reject numbers that cannot be RPPS numbers.

        A Luhn check digit passes for 1 in 10 arbitrary numbers, so a valid
        checksum keeps the pattern score (context words raise it) instead of
        promoting the match to full confidence; an invalid one discards the match.

        :param pattern_text: the text detected by the regex engine
        :return: False if the number is not a valid RPPS number, None otherwise
        """
        if len(pattern_text) != self.RPPS_LENGTH or not pattern_text.isdigit():
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
