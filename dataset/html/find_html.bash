#!/bin/bash
grep -E '"https://[^#"]+' topics.html -o|cut -c2-|grep -v -E '.(js|png|jpg|svg|gif|woff2|php|xml)'|sort|uniq|xargs -P $(nproc||8) wget -T 2 -P downloaded/ -U "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
(cd downloaded/&&ls|xargs -I{} file -i {}|grep -v text/html|cut -f1 -d':'|xargs -P $(nproc||8) rm -v)
#find downloaded/ -type f -exec sha1sum {} +|sort|uniq -d 