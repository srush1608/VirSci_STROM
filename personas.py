

class Persona:
    def __init__(self, name, role, specialization, background, personality):
        self.name = name
        self.role = role
        self.specialization = specialization
        self.background = background
        self.personality = personality

class S0(Persona):
    def __init__(self):
        super().__init__(
            name="S0",
            role="Leader Scientist",
            specialization="High-level research coordination",
            background="PhD in interdisciplinary research, 20+ years of experience",
            personality="Strategic thinker, natural leader, system-oriented"
        )

class S1(Persona):
    def __init__(self):
        super().__init__(
            name="S1",
            role="Historical Context Expert",
            specialization="Historical analysis and context",
            background="PhD in History, specializing in social changes and cultural trends",
            personality="Detail-oriented, analytical, logical thinker"
        )

class S2(Persona):
    def __init__(self):
        super().__init__(
            name="S2",
            role="Technical Insights Expert",
            specialization="Technical analysis and scientific insights",
            background="MSc in Computer Science, specialization in machine learning and AI",
            personality="Problem-solver, practical, and innovative"
        )

class S3(Persona):
    def __init__(self):
        super().__init__(
            name="S3",
            role="Ethical and Societal Impact Expert",
            specialization="Ethical considerations, societal impacts",
            background="PhD in Sociology, with a focus on ethics and social impact",
            personality="Empathetic, thoughtful, ethical decision-maker"
        )
