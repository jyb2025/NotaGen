GENRE_LABELS = [
    'classical',    # 古典（含巴洛克、文艺复兴、中世纪、浪漫主义）
    'jazz',         # 爵士
    'pop',          # 流行
    'folk',         # 民谣（含乡村、蓝草、凯尔特）
    'electronic',   # 电子（含 EDM、New Age、迪斯科）
    'blues',        # 蓝调（含 Soul、R&B）
    'rock',         # 摇滚（含金属）
    'hiphop',       # 嘻哈（含说唱）
    'latin',        # 拉丁（含探戈、桑巴、弗拉门戈、雷鬼）
    'christian',    # 基督教音乐
    'children',     # 儿童音乐
    'epic',         # 史诗音乐
    'other',        # 其他（无法归类的，含 20th_century、contemporary、worldmusic 等）
]

INSTRUMENT_LABELS = [
    'piano',        # 钢琴（含键盘、大键琴）
    'guitar',       # 吉他（含原声、电吉他）
    'voice',        # 人声（含声乐、各声部）
    'strings',      # 弦乐（含小提琴、中提琴、大提琴、低音提琴）
    'woodwinds',    # 木管（含长笛、单簧管、双簧管、巴松管、萨克斯）
    'brass',        # 铜管（含小号、圆号、长号、大号）
    'percussion',   # 打击乐（含鼓）
    'synth',        # 合成器
    'ensemble',     # 合奏（含二重奏、三重奏、四重奏、管弦乐队、室内乐）
]

EMOTION_LABELS = [
    'happy',        # 快乐（含嬉戏、欢乐）
    'sad',          # 悲伤（含忧郁、怀旧、哀悼）
    'calm',         # 平静（含宁静、温柔、优雅、安详）
    'energetic',    # 充满活力（含激进、强烈、热情）
    'dramatic',     # 戏剧性（含紧张、庄重、恐怖）
    'mysterious',   # 神秘（含梦幻）
    'romantic',     # 浪漫（含爱意）
    'heroic',       # 英雄（含史诗、胜利）
    'neutral',      # 中性（占位符，含适中）
]

TEMPO_LABELS = [
    'very_slow',    # 极慢（Largo、Grave，BPM < 60）
    'slow',         # 慢（Adagio、Andante，BPM 60-80）
    'medium',       # 中速（Moderato，BPM 80-120）
    'fast',         # 快（Allegro、Vivace，BPM 120-160）
    'very_fast',    # 极快（Presto，BPM > 160）
]
