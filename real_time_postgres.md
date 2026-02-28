
## SSE(SSE = Server-Sent Events: サーバー送信イベント)
`サーバー → ブラウザ へ一方通行でリアルタイム通知を送る仕組み`


### なにができるのか？
- データが変わったら 自動でフロントに通知 できる
- フロント側は イベントストリームを受け取って UI を更新できる
- WebSocketよりも設定がシンプルで、サーバーからの通知だけなら SSE のほうが楽！

### 具体例 Nuxt3 + Flask + PostgreSQL の構成だと…
1. PostgreSQL で行が INSERT / UPDATE / DELETE される
1. トリガーで pg_notify() が呼ばれる
1. Flask が LISTEN して受け取る
1. Flask が SSE で Nuxt3 に通知する
1. Nuxt3 が自動更新する

### PostgreSQLの設定
```
create table emails (
id SERIAL,
name text,
email text,
primary key(id)
 ) ;

```

```
-- 通知用関数
CREATE OR REPLACE FUNCTION notify_email_change()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('emails_channel', TG_OP);
  IF (TG_OP = 'DELETE') THEN
    RETURN OLD;
  ELSE
    RETURN NEW;
  END IF;
END;
$$ LANGUAGE plpgsql;


-- トリガー設定
CREATE TRIGGER emails_notify_trigger
AFTER INSERT OR UPDATE OR DELETE ON emails
FOR EACH ROW EXECUTE FUNCTION notify_email_change();



---PostgreSQL 10 用
CREATE TRIGGER emails_notify_trigger
AFTER INSERT OR UPDATE OR DELETE ON emails
FOR EACH ROW EXECUTE PROCEDURE notify_email_change();
```  

```
--- 既に存在するトリガーを削除する場合
DROP TRIGGER IF EXISTS emails_notify_trigger ON emails;
```  

## python側のSSEプログラム
```python
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import psycopg2
from psycopg2.extras import RealDictCursor
import select
import json

app = Flask(__name__)
CORS(app)
api = Api(app)

# DB接続
def get_connection():
    return psycopg2.connect(
        dbname="app_01",
        user="postgres",
        password="tankei001",
        host="localhost",
        port="5432"
    )

# emails 全件取得
def fetch_emails():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM emails ORDER BY id;")
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result

# ---------------------------
#   SSE イベントストリーム
# ---------------------------
def event_stream():
    conn = get_connection()
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("LISTEN emails_channel;") # チャンネルを LISTEN

    last_data = None

    while True:
        if select.select([conn], [], [], 10) == ([], [], []): # タイムアウト10秒
            continue

        conn.poll()

        while conn.notifies:
            notify = conn.notifies.pop(0)

            data = fetch_emails()
            if data != last_data:
                last_data = data
                yield f"data: {json.dumps(data)}\n\n"

# SSEストリーム
class EmailsStream(Resource):
    def get(self):
        """GET /emails/stream"""
        return Response(event_stream(), mimetype="text/event-stream") 
        #mimetype="text/event-stream" ⇒ SSE（Server-Sent Events）用のストリーム
        ##片方向のリアルタイム通信 を継続的に送りつづける機能 これを付けないとリアルタイムで受信されない　　

api.add_resource(EmailsStream, "/emails/stream")
if __name__ == "__main__":
    app.run(host="localhost", port=5000, threaded=True, debug=True)
```

## Nuxt側のプログラム
```javascript

<script setup>
    definePageMeta({
      ssr: false   // これがないと hydration mismatch が発生する
    })

    import { ref, onMounted } from "vue"
    const emails = ref([])

    // -------------------------------
    // SSE リアルタイム反映
    // -------------------------------
    onMounted(() => {
      loadData()

      const stream = new EventSource(`${apiBase}/emails/stream`)

      stream.onmessage = (event) => {
        emails.value = JSON.parse(event.data)
      }
    })
</script>
```

