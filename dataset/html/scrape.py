#!/usr/bin/env python3

import http.client
import os
import socket
import sys
import urllib.error
import urllib.request

assert len(sys.argv) > 1

FILEPATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
EARLY_STOP = int(sys.argv[3]) if len(sys.argv) > 3 else None

with open(FILEPATH, 'r', encoding='utf-8') as fp:

	url_list = fp.readlines()

for i, url in enumerate(url_list):

	if EARLY_STOP is not None and i >= EARLY_STOP:

		break

	try:

		with urllib.request.urlopen('http://' + url, timeout=3) as response:

			html = response.read()

			encoding = response.headers.get_content_charset()
			if encoding is None:

				encoding = 'utf-8'

			if encoding not in ('utf-8', 'iso-8859-1'):

				print(url, 'unhandled encoding', encoding)
				continue

			with open(os.path.join(OUTPUT_DIR, f'file_{i}.bin'), 'w', encoding=encoding) as fp:

				fp.write(html.decode(encoding))

	except urllib.error.HTTPError as e:

		print(url, e.code)
	except (http.client.IncompleteRead, urllib.error.URLError, UnicodeError, socket.timeout) as e:

		print(url, e)
