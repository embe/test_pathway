import os

import pathway as pw

HTTP_HOST = os.environ.get("PATHWAY_REST_CONNECTOR_HOST", "127.0.0.1")
HTTP_PORT = os.environ.get("PATHWAY_REST_CONNECTOR_PORT", "8080")


def logic(queries: pw.Table) -> pw.Table:
    return queries.select(
        query_id=queries.id, result=pw.apply(lambda x: x.upper(), pw.this.query)
    )


queries, response_writer = pw.io.http.rest_connector(
    host=HTTP_HOST, port=int(HTTP_PORT)
)
responses = logic(queries)
response_writer(responses)

pw.run_all(debug=True)
