"""Problem-type taxonomy: the coarse "what kind of problem is this" layer.

Exactly one problem_type per problem, always from this whitelist. It is the
partner of the fine-grained concept whitelist in `tutor/knowledge/concepts.py`
(loaded from seeds/concepts.json):

    problem_type  큰 유형, 정확히 1개   e.g. "counting"
    concepts      풀이에 필요한 세부 개념, 0~4개
                  e.g. ["permutation", "permutation_with_identical_elements"]

Solution *strategies* (alternating_arrangement, case_split, ...) belong to
neither: they are how you solve it, not what curriculum knowledge it needs.
If we ever want them they get their own `strategy_tags` field.
"""

from __future__ import annotations

# id → Korean name. Ordered by curriculum area so the prompt reads as a menu.
# 문제 큰 유형 화이트리스트(id → 한국어 이름). 문제 하나에 정확히 하나.
PROBLEM_TYPE_NAMES: dict[str, str] = {
    # 수와 연산
    "arithmetic": "사칙연산 계산",
    "addition_subtraction": "덧셈과 뺄셈",
    "multiplication": "곱셈",
    "division": "나눗셈",
    "fraction": "분수",
    "decimal": "소수",
    "number_theory": "약수·배수·소인수분해",
    "ratio_proportion": "비와 비례",
    "exponent_root": "지수와 제곱근",
    "complex_number": "복소수",
    # 문자와 식
    "algebraic_expression": "문자와 식·다항식",
    "factorization": "인수분해",
    "equation": "방정식 (일반)",
    "linear_equation": "일차방정식",
    "quadratic_equation": "이차방정식",
    "system_of_equations": "연립방정식",
    "identity_remainder": "항등식과 나머지정리",
    "inequality": "부등식",
    # 함수
    "function": "함수 (일반)",
    "linear_function": "일차함수",
    "quadratic_function": "이차함수",
    "sequence": "수열",
    # 기하와 측정
    "geometry": "도형의 성질",
    "coordinate_geometry": "좌표평면과 도형",
    "measurement": "길이·넓이·부피·단위",
    "trigonometry": "삼각비·삼각함수",
    # 자료와 가능성
    "counting": "경우의 수",
    "probability": "확률",
    "statistics": "통계",
    "data_handling": "표와 그래프·자료 정리",
    # 미적분
    "derivative": "미분",
    "integral": "적분",
    "limit": "극한",
    # 기타
    "set_logic": "집합과 명제",
    "word_problem": "문장제 (유형 특정 어려움)",
    "unknown": "분류할 수 없음",
}

# 허용된 유형 id 집합.
ALLOWED_PROBLEM_TYPES = frozenset(PROBLEM_TYPE_NAMES)

# 분류 실패 시 값.
UNKNOWN_PROBLEM_TYPE = "unknown"


# 화이트리스트에 있는 유형인지.
def is_allowed_problem_type(problem_type: str) -> bool:
    return problem_type in ALLOWED_PROBLEM_TYPES


# 목록에 없는 값은 전부 unknown으로 접는다.
def normalize_problem_type(problem_type: str | None) -> str:
    """Anything off the whitelist (or invented) collapses to "unknown"."""
    if problem_type and problem_type.strip() in ALLOWED_PROBLEM_TYPES:
        return problem_type.strip()
    return UNKNOWN_PROBLEM_TYPE


# 프롬프트에 넣을 유형 메뉴.
def problem_types_for_prompt() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in PROBLEM_TYPE_NAMES.items())
