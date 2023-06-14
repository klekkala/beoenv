  import csv

  #params -> filename, sec_gap
  #return -> list of strings
  def read_csv(file_name, sec_gap=3):
    line_count = 0
    content = ""
    csv_reader = csv.reader(file_name, delimiter=',')
    prev = 0
    retlist = []
    for row in csv_reader:
          start = float(row[1])
          end = float(row[2])
          content += str(row[0])
          content += " "
          if start - prev > sec_gap:
            retlist.append(content)
            content = ""
            prev = start
    
    return retlist




  for e in transcripts:
    speaker_name = 'Unknown Speaker'

    for speaker in speakers:
      if e['speaker_id'] == speaker['id']:
        speaker_name = speaker['speaker_name']

    if 'vaisesika' in unidecode(speaker_name).lower():
        speaker_name = 'H.G. Vaiśeṣika Dāsa'

    if speaker_name != current_speaker:
      current_speaker = speaker_name
    else:
      speaker_name = ''

    transcript_content = e['transcript']

    element = {}
    element['type'] = 'paragraph'
    element['anchor'] = str(anchor)
    if len(speaker_name) != 0:
      # print('vaisesika', unidecode(speaker_name).lower(), ('vaisesika' in unidecode(speaker_name)))
      element['speaker'] = speaker_name
    element['content'] = html.escape(transcript_content)

    content.append(element)
    anchor += 1

  return content