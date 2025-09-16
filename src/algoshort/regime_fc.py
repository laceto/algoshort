    def hilo_alternation(self, 
                        hilo: pd.Series, 
                        dist: Optional[pd.Series] = None, 
                        hurdle: Optional[float] = None) -> pd.Series:
        """
        Reduces a series to a succession of highs & lows by eliminating consecutive 
        same-side extremes and keeping only the most extreme values.
        
        This method eliminates same-side consecutive highs and lows where:
        - Highs are assigned a minus sign 
        - Lows are assigned a positive sign
        - The most extreme value is kept when duplicates exist
        
        Parameters:
        -----------
        hilo : pd.Series
            Series containing high/low values with appropriate signs 
            (negative for highs, positive for lows)
        dist : pd.Series, optional
            Distance series for noise filtering
        hurdle : float, optional
            Threshold for noise filtering based on distance
            
        Returns:
        --------
        pd.Series
            Reduced series with alternating highs and lows
            
        Raises:
        -------
        ValueError
            If input data is invalid or empty
        TypeError
            If hilo is not a pandas Series
        """
        try:
            self.logger.debug(f"Starting hilo_alternation with {len(hilo)} data points")
            
            # Input validation
            if not isinstance(hilo, pd.Series):
                raise TypeError("hilo must be a pandas Series")
            
            if hilo.empty:
                self.logger.warning("Empty hilo series provided")
                return pd.Series(dtype=float)
            
            # Create a copy to avoid modifying original data
            hilo_work = hilo.copy()
            initial_length = len(hilo_work.dropna())
            
            i = 0
            max_iterations = 4  # Prevent infinite loops
            
            while (np.sign(hilo_work.shift(1)) == np.sign(hilo_work)).any():
                self.logger.debug(f"Iteration {i+1}: Processing {len(hilo_work.dropna())} points")
                
                # Remove swing lows > swing highs
                condition1 = ((np.sign(hilo_work.shift(1)) != np.sign(hilo_work)) &  
                             (hilo_work.shift(1) < 0) &  
                             (np.abs(hilo_work.shift(1)) < np.abs(hilo_work)))
                hilo_work.loc[condition1] = np.nan
                
                # Remove swing highs < swing lows
                condition2 = ((np.sign(hilo_work.shift(1)) != np.sign(hilo_work)) &  
                             (hilo_work.shift(1) > 0) &  
                             (np.abs(hilo_work) < hilo_work.shift(1)))
                hilo_work.loc[condition2] = np.nan
                
                # Remove duplicate swings (keep extremes) - backward looking
                condition3 = ((np.sign(hilo_work.shift(1)) == np.sign(hilo_work)) & 
                             (hilo_work.shift(1) < hilo_work))
                hilo_work.loc[condition3] = np.nan
                
                # Remove duplicate swings (keep extremes) - forward looking  
                condition4 = ((np.sign(hilo_work.shift(-1)) == np.sign(hilo_work)) & 
                             (hilo_work.shift(-1) < hilo_work))
                hilo_work.loc[condition4] = np.nan
                
                # Apply distance-based noise filtering if provided
                if pd.notnull(dist).any() and hurdle is not None:
                    try:
                        distance_condition = ((np.sign(hilo_work.shift(1)) != np.sign(hilo_work)) &
                                            (np.abs(hilo_work + hilo_work.shift(1)).div(dist, fill_value=1) < hurdle))
                        hilo_work.loc[distance_condition] = np.nan
                        self.logger.debug(f"Applied distance filtering with hurdle={hurdle}")
                    except Exception as e:
                        self.logger.warning(f"Distance filtering failed: {e}")
                
                # Reduce hilo after each pass
                hilo_work = hilo_work.dropna().copy()
                
                i += 1
                if i >= max_iterations:
                    self.logger.warning(f"Maximum iterations ({max_iterations}) reached in hilo_alternation")
                    break
            
            final_length = len(hilo_work)
            reduction_ratio = (initial_length - final_length) / initial_length * 100 if initial_length > 0 else 0
            
            self.logger.info(f"hilo_alternation completed: reduced from {initial_length} to {final_length} "
                           f"points ({reduction_ratio:.1f}% reduction) in {i} iterations")
            
            return hilo_work
            
        except Exception as e:
            self.logger.error(f"Error in hilo_alternation: {e}")
            raise