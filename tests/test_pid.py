"""Tests for PID module."""

import pytest


class TestProposer:
    """Test proposer functionality."""

    def test_proposer_creation(self):
        from npc_gym.pid.proposer import Proposer, ProposerConfig

        config = ProposerConfig(name="test_proposer", domain="reasoning")
        proposer = Proposer(config=config)

        assert proposer.id == "test_proposer"
        assert proposer.config.domain == "reasoning"

    def test_proposal_structure(self):
        from npc_gym.pid.proposer import Proposal

        proposal = Proposal(
            content="This is a hypothesis",
            confidence=0.8,
            proposer_id="test",
            info_used=["fragment1", "fragment2"],
        )

        assert proposal.confidence == 0.8
        assert len(proposal.info_used) == 2
        assert str(proposal) == "[0.80] This is a hypothesis"

    def test_proposer_ensemble(self):
        from npc_gym.pid.proposer import ProposerEnsemble, Proposer, ProposerConfig

        ensemble = ProposerEnsemble()

        for i in range(3):
            config = ProposerConfig(name=f"proposer_{i}")
            ensemble.add_proposer(Proposer(config=config))

        assert len(ensemble.proposers) == 3
        assert "proposer_0" in ensemble.proposers


class TestVoter:
    """Test voter functionality."""

    def test_voter_creation(self):
        from npc_gym.pid.voter import Voter, VoterConfig

        config = VoterConfig(name="test_voter")
        voter = Voter(config=config)

        assert voter.id == "test_voter"

    def test_vote_structure(self):
        from npc_gym.pid.voter import Vote

        vote = Vote(
            voter_id="test",
            rankings=[1, 0, 2],
            scores=[0.3, 0.5, 0.2],
            confidence=0.7,
        )

        assert vote.rankings[0] == 1  # First choice is proposal 1
        assert vote.confidence == 0.7

    def test_voting_ensemble(self):
        from npc_gym.pid.voter import VotingEnsemble, Voter, VoterConfig, VotingStrategy

        ensemble = VotingEnsemble(strategy=VotingStrategy.WEIGHTED)

        for i in range(2):
            config = VoterConfig(name=f"voter_{i}")
            ensemble.add_voter(Voter(config=config), weight=1.0)

        assert len(ensemble.voters) == 2

    def test_heuristic_voting(self):
        from npc_gym.pid.voter import Voter, VoterConfig
        from npc_gym.pid.proposer import Proposal

        voter = Voter(config=VoterConfig(name="test"))

        proposals = [
            Proposal(content="A", confidence=0.8, proposer_id="p1", info_used=[]),
            Proposal(content="B", confidence=0.3, proposer_id="p2", info_used=[]),
            Proposal(content="C", confidence=0.6, proposer_id="p3", info_used=[]),
        ]

        vote = voter.vote(proposals)

        # Highest confidence should be ranked first
        assert vote.rankings[0] == 0  # Proposal A has highest confidence


class TestSynthesizer:
    """Test synthesizer functionality."""

    def test_synthesizer_creation(self):
        from npc_gym.pid.synthesizer import Synthesizer, SynthesizerConfig

        config = SynthesizerConfig(name="test_synth")
        synth = Synthesizer(config=config)

        assert synth.id == "test_synth"

    def test_synthesis_best_only(self):
        from npc_gym.pid.synthesizer import Synthesizer, SynthesizerConfig, SynthesisStrategy
        from npc_gym.pid.proposer import Proposal

        config = SynthesizerConfig(name="synth", strategy=SynthesisStrategy.BEST_ONLY)
        synth = Synthesizer(config=config)

        proposals = [
            Proposal(content="Best", confidence=0.9, proposer_id="p1", info_used=[]),
            Proposal(content="Worst", confidence=0.2, proposer_id="p2", info_used=[]),
        ]

        result = synth.synthesize(proposals)
        assert result.content == "Best"

    def test_synthesis_consensus(self):
        from npc_gym.pid.synthesizer import Synthesizer, SynthesizerConfig, SynthesisStrategy
        from npc_gym.pid.proposer import Proposal

        config = SynthesizerConfig(name="synth", strategy=SynthesisStrategy.CONSENSUS)
        synth = Synthesizer(config=config)

        proposals = [
            Proposal(content="The fox is quick", confidence=0.7, proposer_id="p1", info_used=[]),
            Proposal(content="A quick fox", confidence=0.6, proposer_id="p2", info_used=[]),
            Proposal(content="Quick animal fox", confidence=0.5, proposer_id="p3", info_used=[]),
        ]

        result = synth.synthesize(proposals)
        # "quick" and "fox" should be in consensus
        assert "quick" in result.content.lower() or "fox" in result.content.lower()


class TestOptimizer:
    """Test PID optimizer."""

    def test_optimizer_creation(self):
        from npc_gym.pid.optimizer import PIDOptimizer, OptimizationConfig

        config = OptimizationConfig(num_epochs=10)
        optimizer = PIDOptimizer(config=config)

        assert optimizer.config.num_epochs == 10

    def test_info_efficiency_metric(self):
        from npc_gym.pid.optimizer import InfoEfficiencyMetric

        metric = InfoEfficiencyMetric(
            total_info_available=10,
            info_used=3,
            accuracy_achieved=0.8,
            time_to_answer=1.5,
        )

        assert metric.efficiency_score > 0
        assert metric.bits_per_accuracy > 0

        # More efficient than using all info
        metric2 = InfoEfficiencyMetric(
            total_info_available=10,
            info_used=10,
            accuracy_achieved=0.8,
            time_to_answer=1.5,
        )

        assert metric.efficiency_score > metric2.efficiency_score

    def test_create_pid_system(self):
        from npc_gym.pid.optimizer import create_pid_system

        proposers, voters, synth, optimizer = create_pid_system(num_proposers=2)

        assert len(proposers.proposers) == 2
        assert len(voters.voters) == 2
        assert synth is not None
        assert optimizer.proposers is not None


class TestIntegration:
    """Integration tests for PID system."""

    def test_full_pipeline_mock(self):
        """Test full PID pipeline without actual LLM calls."""
        from npc_gym.pid.proposer import ProposerEnsemble, Proposer, ProposerConfig, Proposal
        from npc_gym.pid.voter import VotingEnsemble, Voter, VoterConfig
        from npc_gym.pid.synthesizer import Synthesizer, SynthesizerConfig, SynthesisStrategy

        # Create ensemble
        proposers = ProposerEnsemble()
        for i in range(3):
            proposers.add_proposer(Proposer(ProposerConfig(name=f"p{i}")))

        voters = VotingEnsemble()
        voters.add_voter(Voter(VoterConfig(name="v1")))

        synth = Synthesizer(SynthesizerConfig(name="synth", strategy=SynthesisStrategy.BEST_ONLY))

        # Mock proposals (since no LLM)
        proposals = [
            Proposal(content="Answer A", confidence=0.7, proposer_id="p0", info_used=["f1"]),
            Proposal(content="Answer B", confidence=0.9, proposer_id="p1", info_used=["f2"]),
            Proposal(content="Answer C", confidence=0.5, proposer_id="p2", info_used=["f3"]),
        ]

        # Vote
        vote_result = voters.vote(proposals)
        assert "winner_idx" in vote_result

        # Synthesize
        result = synth.synthesize(proposals, vote_result)
        assert result.content == "Answer B"  # Highest confidence
