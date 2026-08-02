"""Echo mode simulates worksheet progression: each DISTINCT image advances a stage."""

from tutor.llm.echo import EchoLLMClient
from tutor.vision.recognizer import Recognition


def recognize(llm: EchoLLMClient, image: bytes) -> Recognition:
    return llm.complete_json(
        purpose="recognize", system="", user="", images=[image], schema=Recognition
    )


def test_distinct_images_advance_stages():
    llm = EchoLLMClient()
    assert recognize(llm, b"img-a").student_work == ["3*x = 20 + 5"]
    # same image again → same stage (repeated hint on unchanged work escalates)
    assert recognize(llm, b"img-a").student_work == ["3*x = 20 + 5"]
    # new image = the student wrote more → next stage
    assert recognize(llm, b"img-b").student_work == ["3*x = 15"]
    assert recognize(llm, b"img-c").student_work == ["3*x = 15", "x = 5"]
    # beyond the script, further images stay on the last stage
    assert recognize(llm, b"img-d").student_work == ["3*x = 15", "x = 5"]
    # going back to a seen image returns its original stage
    assert recognize(llm, b"img-a").student_work == ["3*x = 20 + 5"]


def test_scripted_queue_still_wins():
    llm = EchoLLMClient({"recognize": [{"problem_text": "p", "student_work": ["custom"]}]})
    assert recognize(llm, b"img-a").student_work == ["custom"]
