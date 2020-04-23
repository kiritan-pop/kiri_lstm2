# coding: utf-8
require 'nokogiri'
require 'pp'
require 'sqlite3'


# --- debug switch  true false
VERB = false

############################################################
#メイン処理
marge_text = ""
mediatext1b = ""
username = ''
i = 0
db = SQLite3::Database.new('db/nico_statuses.db',:timeout=>1200)
f_toot = File.open('tmp/toot_nico.txt', "w")
f_ids = File.open('tmp/ids_nico.txt', "w")

sql = "select id,content,acct from statuses order by id desc"
db.execute(sql)  { |id,content,acct|
    break if i > 100000
    next if acct=='kiri_bot01'
    contents = Nokogiri::HTML.parse(content)
    text = ''
    has_tag = false
    contents.search('p').children.each{|item|
        text += item.text.strip  if item.text?
    }
    f_toot.puts(text)
    f_ids.puts('nico_'+id.to_s)
    i += 1
}

db.close
f_toot.close
