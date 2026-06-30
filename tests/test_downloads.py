import pytest

from littrace.config import LitTraceConfig
from littrace.downloads import execute_downloads
from littrace.models import AccessType, DownloadExecutionRequest, PaperMetadata


@pytest.mark.anyio
async def test_execute_downloads_returns_login_action_for_gated_papers():
    result = await execute_downloads(
        LitTraceConfig(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Login paper",
                access_type=AccessType.REQUIRES_LOGIN,
                source_urls=["https://example.org/login-paper"],
            )
        ],
        DownloadExecutionRequest(dry_run=True),
    )

    assert result.requires_login_count == 1
    assert result.items[0].action == "open_login_popup"
    assert result.items[0].target_path
    assert result.items[0].login_instructions
    assert "allowed to use" in result.items[0].login_instructions[1]
