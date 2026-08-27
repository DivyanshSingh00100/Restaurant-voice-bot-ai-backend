import pytest

from app.guardrails.base_guardrail import BaseGuardrail


def test_base_guardrail_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseGuardrail()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_subclass_implementing_check_can_be_instantiated():
    class DummyGuardrail(BaseGuardrail):
        async def check(self, text: str) -> bool:
            return True

    guardrail = DummyGuardrail()
    assert await guardrail.check("hello") is True
