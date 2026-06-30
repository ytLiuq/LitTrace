import pytest
import httpx

from littrace.cache import cache_key, write_text_cache
from littrace.citations import check_link, citation_for_paper
from littrace.config import LitTraceConfig, StorageConfig
from littrace.models import LinkStatus, PaperMetadata


@pytest.mark.anyio
async def test_check_link_uses_cached_record(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(cache_dir=tmp_path / "cache"))
    record = citation_for_paper(PaperMetadata(paper_id="p1", title="Paper", doi="10.1000/example"))
    record.link_status = LinkStatus.VERIFIED_200
    write_text_cache(config, "citation_links", cache_key(str(record.access_url)), record.model_dump_json())

    async with httpx.AsyncClient() as client:
        checked = await check_link(client, record, config)

    assert checked.link_status == LinkStatus.VERIFIED_200
