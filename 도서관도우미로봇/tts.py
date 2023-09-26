def synthesize_text(text):
    import io
    from google.cloud import texttospeech
    from pydub import AudioSegment
    from pydub.playback import play
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code='ko-KR',  # 원하는 언어 코드로 변경
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE  # 음성 성별 설정 (NEUTRAL, MALE, FEMALE 중 선택)
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3  # 출력 오디오 인코딩 형식 (MP3, LINEAR16, WAV 등)
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    audio_content = response.audio_content

    # 오디오 데이터를 재생 가능한 형식으로 변환
    audio_stream = io.BytesIO(audio_content)
    audio_segment = AudioSegment.from_file(audio_stream, format="mp3")

    # 음성 출력
    play(audio_segment)

# 텍스트를 음성으로 변환하고 바로 출력
# synthesize_text(text)
