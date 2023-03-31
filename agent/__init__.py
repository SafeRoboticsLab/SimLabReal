from agent.policy_prior_both_latent import PolicyPriorBothLatent
from agent.policy_prior_perf_latent import PolicyPriorPerfLatent
from agent.policy_prior_shield import PolicyPriorShield
from agent.policy_prior_base_pac import PolicyPriorBasePAC
from agent.policy_prior_stack_shield import PolicyPriorStackShield
from agent.policy_prior_stack_perf_latent import PolicyPriorStackPerfLatent
from agent.policy_prior_stack_base_pac import PolicyPriorStackBasePAC
from agent.policy_prior_stack_both_latent import PolicyPriorStackBothLatent

from agent.policy_posterior_both_latent import PolicyPosteriorBothLatent
from agent.policy_posterior_perf_latent import PolicyPosteriorPerfLatent
from agent.policy_posterior_shield import PolicyPosteriorShield
from agent.policy_posterior_base_pac import PolicyPosteriorBasePAC
from agent.policy_posterior_stack_both_latent import PolicyPosteriorStackBothLatent
from agent.policy_posterior_stack_perf_latent import PolicyPosteriorStackPerfLatent
from agent.policy_posterior_stack_shield import PolicyPosteriorStackShield
from agent.policy_posterior_stack_base_pac import PolicyPosteriorStackBasePAC

from agent.naive_rl import NaiveRL
from agent.naive_stack_rl import NaiveStackRL

from agent.recovery_rl import RecoveryRL
from agent.recovery_rl_stack import RecoveryRLStack
from agent.sqrl_pre import SQRLPre
from agent.sqrl_pre_stack import SQRLPreStack
from agent.sqrl_fine import SQRLFine
from agent.sqrl_fine_stack import SQRLFineStack

agent_dict = {
    'PolicyPriorBothLatent': PolicyPriorBothLatent,
    'PolicyPriorPerfLatent': PolicyPriorPerfLatent,
    'PolicyPriorShield': PolicyPriorShield,
    'PolicyPriorStackBothLatent': PolicyPriorStackBothLatent,
    'PolicyPriorStackPerfLatent': PolicyPriorStackPerfLatent,
    'PolicyPriorStackShield': PolicyPriorStackShield,
    'PolicyPriorBasePAC': PolicyPriorBasePAC,
    'PolicyPriorStackBasePAC': PolicyPriorStackBasePAC,
    'SQRLPre': SQRLPre,
    'SQRLPreStack': SQRLPreStack,
    #
    'NaiveRL': NaiveRL,
    'NaiveStackRL': NaiveStackRL,
    #
    'PolicyPosteriorBothLatent': PolicyPosteriorBothLatent,
    'PolicyPosteriorPerfLatent': PolicyPosteriorPerfLatent,
    'PolicyPosteriorShield': PolicyPosteriorShield,
    'PolicyPosteriorStackBothLatent': PolicyPosteriorStackBothLatent,
    'PolicyPosteriorStackPerfLatent': PolicyPosteriorStackPerfLatent,
    'PolicyPosteriorStackShield': PolicyPosteriorStackShield,
    'PolicyPosteriorBasePAC': PolicyPosteriorBasePAC,
    'PolicyPosteriorStackBasePAC': PolicyPosteriorStackBasePAC,
    'RecoveryRL': RecoveryRL,
    'RecoveryRLStack': RecoveryRLStack,
    'SQRLFine': SQRLFine,
    'SQRLFineStack': SQRLFineStack,
}
