import base64
import dill
from prefect.serializers import Serializer


class DillSerializer(Serializer):
    type = "dill"

    def dumps(self, obj) -> bytes:
        blob = dill.dumps(obj)
        return base64.encodebytes(blob)

    def loads(self, blob: bytes):
        return dill.loads(base64.decodebytes(blob))
