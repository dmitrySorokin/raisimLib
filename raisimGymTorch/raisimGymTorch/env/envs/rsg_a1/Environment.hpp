//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

/* Convention
*
*   observation space = [ height,                       n =  1, si (start index) =  0
*                         body roll,,                   n =  1, si =  1
*                         body pitch,,                  n =  1, si =  2
*                         joint angles,                 n = 12, si =  3
*                         body Linear velocities,       n =  3, si = 15
*                         body Angular velocities,      n =  3, si = 18
*                         joint velocities,             n = 12, si = 21
*                         contacts binary vector,       n =  4, si = 33
*                         previous action,              n = 12, si = 37 ] total 49
*
*   action space      = [ joint angles                  n = 12, si =  0 ] total 12
*/

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    // Load configuration
    setSimulationTimeStep(cfg["simulation_dt"].template As<double>());
    setControlTimeStep(cfg["control_dt"].template As<double>());
    k_c = cfg["k_0"].template As<double>();
    k_d = cfg["k_d"].template As<double>();
    // ...


    /// add objects
    a1_ = world_->addArticulatedSystem(resourceDir_ + "/a1/urdf/a1.urdf");
    a1_->setName("Unitree A1");
    a1_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    // Terrain
    // auto ground = world_->addGround(); // Flat terrain
    raisim::TerrainProperties terrainProperties; // Randomized terrain
    terrainProperties.frequency = cfg["terrain"]["frequency"].template As<double>();
    terrainProperties.zScale    = cfg["terrain"]["zScale"].template As<double>();
    terrainProperties.xSize     = cfg["terrain"]["xSize"].template As<double>();
    terrainProperties.ySize     = cfg["terrain"]["ySize"].template As<double>();
    terrainProperties.xSamples  = cfg["terrain"]["xSamples"].template As<size_t>();
    terrainProperties.ySamples  = cfg["terrain"]["ySamples"].template As<size_t>();
    terrainProperties.fractalOctaves    = cfg["terrain"]["fractalOctaves"].template As<size_t>();
    terrainProperties.fractalLacunarity = cfg["terrain"]["fractalLacunarity"].template As<double>();
    terrainProperties.fractalGain       = cfg["terrain"]["fractalGain"].template As<double>();

    auto hm = world_->addHeightMap(0.0, 0.0, terrainProperties);


    /// get robot data
    gcDim_ = a1_->getGeneralizedCoordinateDim();
    gvDim_ = a1_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    // Generate robot initial pose
    x0Dist_ = std::uniform_real_distribution<double>(-1, 1);
    y0Dist_ = std::uniform_real_distribution<double>(-1, 1);

    /// this is nominal standing configuration of unitree A1
    // P_x, P_y, P_z, 1.0, A_x, A_y, A_z, FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf.
    // gc_init_ << 0.0, 0.0, 0.39, 1.0, 0.0, 0.0, 0.0, 0.06, 0.6, -1.2, -0.06, 0.6, -1.2, 0.06, 0.6, -1.2, -0.06, 0.6, -1.2;
    gc_init_ << x0Dist_(randomGenerator_), y0Dist_(randomGenerator_), 0.45, 1.0, 0.0, 0.0, 0.0, 0.06, 0.6, -1.2, -0.06, 0.6, -1.2, 0.06, 0.6, -1.2, -0.06, 0.6, -1.2;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(55.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.8);
    a1_->setPdGains(jointPgain, jointDgain);
    a1_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 49; /// convention described on top
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    groundImpactForces_.setZero();
    previousJointPositions_.setZero(nJoints_);

    /// indices of links that should make contact with ground
    contactIndices_.insert(a1_->getBodyIdx("FL_calf"));
    contactIndices_.insert(a1_->getBodyIdx("FR_calf"));
    contactIndices_.insert(a1_->getBodyIdx("RL_calf"));
    contactIndices_.insert(a1_->getBodyIdx("RR_calf"));

    // Define mapping of body ids to sequential indices
    contactSequentialIndex_[a1_->getBodyIdx("FL_calf")] = 0;
    contactSequentialIndex_[a1_->getBodyIdx("FR_calf")] = 1;
    contactSequentialIndex_[a1_->getBodyIdx("RL_calf")] = 2;
    contactSequentialIndex_[a1_->getBodyIdx("RR_calf")] = 3;

    // Initialize materials
    world_->setMaterialPairProp("default", "rubber", 0.8, 0.15, 0.001);

    // TODO: Move values to config
    // Initialize environmental sampler distributions
    decisionDist_ = std::uniform_real_distribution<double>(0, 1);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(a1_);
    }
  }

  void init() final { }

  void reset() final {
    gc_init_[0] = x0Dist_(randomGenerator_);
    gc_init_[1] = y0Dist_(randomGenerator_);

    a1_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    a1_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    // Record values for next step calculations
    previousJointPositions_ = gc_.tail(nJoints_);

    rewards_.record("torque", a1_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));

    // Apply random force to the COM
    auto applyingForceDecision = decisionDist_(randomGenerator_);
    if (applyingForceDecision < 0.5) {
    	auto externalEffort = 1000 * Eigen::VectorXd::Random(3);
    	a1_->setExternalForce(a1_->getBodyIdx("base"), externalEffort);
    }

    // Apply random torque to the COM
    applyingForceDecision = decisionDist_(randomGenerator_);
    if (applyingForceDecision < 0.05) {
    	auto externalTorque = 100 * Eigen::VectorXd::Random(3);
    	a1_->setExternalTorque(a1_->getBodyIdx("base"), externalTorque);
    }

    return rewards_.sum();
  }

  void updateObservation() {
    a1_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    Eigen::VectorXd contacts;
    contacts.setZero(footContactState_.size());
    for (auto &fs: footContactState_)  fs = false;

    /// Dirive contacts vector
    for (auto &contact: a1_->getContacts()) {
      if (!contact.isSelfCollision() && contact.getPairObjectBodyType() == BodyType::STATIC) {
        if (contactIndices_.find(contact.getlocalBodyIndex()) != contactIndices_.end()) {
          auto normalForce = contact.getContactFrame().e() * contact.getImpulse().e() / world_->getTimeStep();
          auto groundImpactForce = sqrt(pow(normalForce[0],2) + pow(normalForce[1],2) + pow(normalForce[2],2));

          auto bodyIndex = contact.getlocalBodyIndex();
          groundImpactForces_[contactSequentialIndex_[bodyIndex]] = groundImpactForce;
          footContactState_[contactSequentialIndex_[bodyIndex]] = true;
          contacts[contactSequentialIndex_[bodyIndex]] = 1;
        }
      }
    }

    // Nosify observations
    auto velocitiesNoised = gv_.tail(nJoints_) + 0.5 * Eigen::VectorXd::Random(nJoints_);

    auto bodyLinearVelocityNoised = bodyLinearVel_ + 0.08 * Eigen::VectorXd::Random(bodyLinearVel_.size());
    auto bodyAngularVelocityNoised = bodyAngularVel_ + 0.16 * Eigen::VectorXd::Random(bodyAngularVel_.size());

    double euler_angles[3];
    raisim::quatToEulerVec(&gc_[3], euler_angles);

    obDouble_ << gc_[2],                // body height 1
      euler_angles[0], euler_angles[1], // body roll & pitch 2
      gc_.tail(nJoints_),               // joint angles 12
      bodyLinearVelocityNoised,         // body linear 3
      bodyAngularVelocityNoised,        // angular velocity 3
      velocitiesNoised,                 // joint velocity 12
      contacts,                         // contacts binary vector 4
      previousJointPositions_;          // previous action 12
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    // Terminal condition
    double euler_angles[3];
    raisim::quatToEulerVec(&gc_[3], euler_angles);
    if (gc_[2] < 0.28 || fabs(euler_angles[0]) > 0.4 || fabs(euler_angles[1]) > 0.2)  {
        return true;
    }

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* a1_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  Eigen::Vector4d groundImpactForces_;
  Eigen::VectorXd previousJointPositions_;

  // Curriculum factors
  double k_c, k_d;

  std::random_device randomGenerator_;
  std::uniform_real_distribution<double> x0Dist_;
  std::uniform_real_distribution<double> y0Dist_;

  // Random stuff for environmental parameters
  std::uniform_real_distribution<double> decisionDist_;

  // Contacts information
  std::set<size_t> contactIndices_;
  std::array<bool, 4> footContactState_;
  std::unordered_map<int, int> contactSequentialIndex_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

