// Copyright (C) 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpCGPenaltyData.cpp 2276 2013-05-05 12:33:44Z stefan $
//
// Authors:  Andreas Waechter           IBM     2007-06-04
//                   based on IpIpoptData.hpp

#include "IpCGPenaltyData.hpp"

namespace Ipopt
{

  CGPenaltyData::CGPenaltyData()
  {}

  CGPenaltyData::~CGPenaltyData()
  {}

  bool CGPenaltyData::Initialize(const Journalist& jnlst,
                                 const OptionsList& options,
                                 const std::string& prefix)
  {

    have_cgpen_deltas_ = false;
    have_cgfast_deltas_ = false;

    initialize_called_ = true;

    penalty_initialized_ = false;
    kkt_penalty_initialized_ = false;
    have_cgpen_deltas_ = false;
    have_cgfast_deltas_ = false;
    curr_penalty_pert_ = 0.;
    max_alpha_x_ = 1.;
    never_try_pure_Newton_ = false;
    restor_iter_ = -1;
    restor_counter_ = 0.;



    return true;
  }

  bool CGPenaltyData::InitializeDataStructures()
  {
    DBG_ASSERT(initialize_called_);
#if COIN_IPOPT_CHECKLEVEL > 0

    debug_delta_cgpen_tag_ = TaggedObject::Tag();
    debug_delta_cgfast_tag_ = TaggedObject::Tag();
    debug_delta_cgpen_tag_sum_ = TaggedObject::Tag();
    debug_delta_cgfast_tag_sum_ = TaggedObject::Tag();
#endif

    // Set the pointers for storing steps to NULL
    delta_cgpen_ = NULL;
    delta_cgfast_ = NULL;

    have_cgpen_deltas_ = false;
    have_cgfast_deltas_ = false;

    return true;
  }

  void CGPenaltyData::AcceptTrialPoint()
  {
    // Free the memory for the Chen-Goldfarb step
    delta_cgpen_ = NULL;
    delta_cgfast_ = NULL;

    have_cgpen_deltas_ = false;
    have_cgfast_deltas_ = false;
  }

} // namespace Ipopt
